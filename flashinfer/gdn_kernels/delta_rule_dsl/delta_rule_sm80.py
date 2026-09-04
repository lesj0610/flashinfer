import functools
from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp, warpgroup, cpasync
from .alpha import AlphaProcessor
from .collective_store_sm80 import CollectiveStoreSm80
from .pipeline_sm80 import PipelineCpAsyncSm80
from .custom_compile_cache import (
    KeyedCompileMixin,
    cached_compile,
    get_cached_compile,
    sm8x_compile_options,
)
from .collective_inverse_hmma import CollectiveInverse
from .helpers import SM80, round_down, state_dtype_to_cutlass
from .schedule import WorkDesc
from .varlen_helper import is_integer_dtype


@functools.cache
def _sm80_compile_options(device):
    return (cute.EnableTVMFFI(True),) + sm8x_compile_options(device)


# Every thread in the block issues its share of a tile. TMA was one thread and
# needed no such width; cp.async needs the block, because a thread can only
# wait on copies it issued itself.
NUM_MMA_WARP_GROUPS = 2
THREADS_PER_WARP_GROUP = 128
LOAD_THREADS = NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
# bf16 elements in one 16 B cp.async access.
_LOAD_VEC_ELEMS = 8
_LOAD_ALIGN_BYTES = 16


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# Converting to or from FP8 is a single instruction from SM89 on. Earlier SM8x
# parts have no such instruction and no software path in this kernel, so the
# state cannot be held in FP8 there.
_FP8_MIN_CAPABILITY = (8, 9)


def _check_state_dtype_supported(device: torch.device, **tensors) -> None:
    """Reject an FP8 state on the SM8x parts that cannot convert to it.

    Without this the request reaches the compiler and comes back as a bare
    "NVVM backend compilation failed", which says nothing about which argument
    caused it.
    """
    capability = torch.cuda.get_device_capability(device)
    if capability >= _FP8_MIN_CAPABILITY:
        return
    for name, tensor in tensors.items():
        if tensor is not None and tensor.dtype in _FP8_DTYPES:
            raise NotImplementedError(
                f"{name} is {tensor.dtype}, which the GDN prefill kernel cannot "
                f"produce on compute capability {capability[0]}.{capability[1]}: "
                f"FP8 conversion instructions start at "
                f"{_FP8_MIN_CAPABILITY[0]}.{_FP8_MIN_CAPABILITY[1]}"
            )


def _check_load_alignment(**tensors: torch.Tensor) -> None:
    """Reject inputs the vectorized loads would silently misread.

    The loads take the widest cp.async there is, and the kernel states the
    alignment the layout algebra cannot prove by rounding each partitioned
    pointer up to 16 B. That rounding is identity only while the premise holds:
    the base address on a 16 B boundary, and every stride the partition walks a
    multiple of eight elements. A tensor that breaks either would be read from
    the wrong address rather than rejected, so the premise is checked here.

    It holds for anything this kernel is called with -- head_dim is 128 and the
    packed layouts stride by whole heads -- so this is a guard against an
    unusual view arriving, not a case that needs handling.
    """
    for name, t in tensors.items():
        if t is None:
            continue
        if t.data_ptr() % _LOAD_ALIGN_BYTES != 0:
            raise ValueError(
                f"{name} must be {_LOAD_ALIGN_BYTES}-byte aligned for the sm_80 "
                f"delta-rule kernel, got address {t.data_ptr():#x}"
            )
        bad = [
            (dim, s)
            for dim, s in enumerate(t.stride())
            if s != 1 and s % _LOAD_VEC_ELEMS != 0
        ]
        if bad:
            raise ValueError(
                f"{name} strides {t.stride()} are not vectorizable for the sm_80 "
                f"delta-rule kernel: dim(s) {[d for d, _ in bad]} must be a "
                f"multiple of {_LOAD_VEC_ELEMS} elements or contiguous"
            )


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync all 128 WG0 threads before collective_inverse


# The sm_90 kernel also has a WarpGroupRole.LDST and a LoadStoreWarpRole
# splitting that group four ways. Both are gone: handing a tile from a load
# warp to a math warp needs an mbarrier this DSL will not emit for sm_80 (see
# pipeline_sm80), so the whole block does both jobs and only the two math roles
# below remain.
class MathWarpGroupRole(IntEnum):
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads, 1, 1)
# Block: 256 threads → WG0=[0,127], WG1=[128,255]
#
# needs_alpha / needs_beta / needs_init_state are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class _FullyFusedDeltaRuleSm80(KeyedCompileMixin):
    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        mma_registers = round_down(
            (total_registers - load_registers) // num_mma_warp_groups,
            reg_alloc_granularity,
        )
        return min(248, load_registers), min(248, mma_registers)

    @staticmethod
    def can_implement(
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_size: int,
        element_size: int,
    ) -> bool:
        ratio = (
            num_q_heads // num_v_heads
            if num_q_heads > num_v_heads
            else num_v_heads // num_q_heads
        )
        is_gva_enabled = num_v_heads > num_q_heads

        is_gqa_like = (
            (num_k_heads == num_v_heads)
            and (num_q_heads == ratio * num_k_heads)
            and (num_q_heads == ratio * num_v_heads)
        )
        is_gva_like = (
            (num_q_heads == num_k_heads)
            and (num_v_heads == ratio * num_q_heads)
            and (num_v_heads == ratio * num_k_heads)
        )

        alignment = 16 // element_size
        return (
            ((not is_gva_enabled and is_gqa_like) or (is_gva_enabled and is_gva_like))
            and (head_size <= 128)
            and ((head_size % alignment) == 0)
        )

    def __init__(
        self,
        needs_alpha: bool,
        needs_beta: bool,
        needs_init_state: bool,
        needs_checkpointing: bool,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        initial_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        checkpoint_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        use_state_indices: bool = False,
        cu_seqlens_dtype: torch.dtype = torch.int64,
        state_indices_dtype: torch.dtype | None = None,
        checkpoint_cu_starts_dtype: torch.dtype | None = None,
        state_inner_strides: tuple[int, ...] | None = None,
        init_state_inner_strides: tuple[int, ...] | None = None,
    ):
        self.needs_alpha = needs_alpha
        self.needs_beta = needs_beta
        self.needs_init_state = needs_init_state
        self.needs_checkpointing = needs_checkpointing
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.checkpoint_state_dtype = checkpoint_state_dtype
        self.use_state_indices = use_state_indices
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.state_indices_dtype = state_indices_dtype
        self.checkpoint_cu_starts_dtype = checkpoint_cu_starts_dtype
        self.state_inner_strides = state_inner_strides
        self.init_state_inner_strides = init_state_inner_strides
        self.inverse_dtype = cutlass.Float16
        self.BLK_Q = 64
        self.BLK_KV = 64
        self.D = 128
        self.q_stage = 1
        # One stage, not the sm_90 kernel's two. A second buffer only pays if
        # a load can run ahead of the math, and none can: PipelineCpAsyncSm80
        # is constructed with prefetch left at zero everywhere, so the loads
        # commit and the math drains them inside one iteration and the extra
        # 16 KiB was never read from.
        #
        # Freeing it does not buy residency -- that is register-limited at one
        # CTA either way -- so this is a 16 KiB reclaim with no measured
        # regression, not a speedup. It is also the 16 KiB a later look-ahead
        # would need back.
        self.k_stage = 1
        self.v_stage = 1
        self.o_stage = 1
        self.alpha_beta_stage = 2
        # One store object for the kernel: the tile shape and the block width
        # are both fixed at construction.
        self.o_writer = CollectiveStoreSm80(self.BLK_Q, self.D, LOAD_THREADS)
        self.manual_cache_key(
            "needs_alpha",
            "needs_beta",
            "needs_init_state",
            "needs_checkpointing",
            "dtype",
            "acc_dtype",
            "initial_state_dtype",
            "state_dtype",
            "checkpoint_state_dtype",
            "use_state_indices",
            "cu_seqlens_dtype",
            "state_indices_dtype",
            "checkpoint_cu_starts_dtype",
            "state_inner_strides",
            "init_state_inner_strides",
            "inverse_dtype",
            "BLK_Q",
            "BLK_KV",
            "D",
            "q_stage",
            "k_stage",
            "v_stage",
            "o_stage",
            "alpha_beta_stage",
        )

    def get_next_work(
        self,
        cu_seqlens: cute.Tensor,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ) -> WorkDesc:
        bx, _, _ = cute.arch.block_idx()
        seq_idx = bx // num_sab_heads
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        tok_start = cu_seqlens[seq_idx]
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1] - tok_start)

        return WorkDesc(
            seq_idx=seq_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_start,
            seq_len=seq_len,
            tile_idx=cutlass.Int32(0),
        )

    # ─── Ordered 2-WG math barriers ───────────────────────────────────────────
    # Translates flat::OrderedNamedBarriers<UseReservedNB, NB0, NB1>.
    # wg_idx: MathWarpGroupRole.KK or MathWarpGroupRole.QK.

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Pre-arrive at WG0's barrier so WG0 is unblocked on the first wait."""
        if wg_idx == MathWarpGroupRole.QK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        """Arrive+wait on this WG's own ordered barrier."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)
        else:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Arrive at the other WG's barrier to unblock it."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    # ─── kk_store_and_inv ─────────────────────────────────────────────────────

    @cute.jit
    def _kk_store_and_inv(
        self,
        tKKrKK: cute.Tensor,  # fp32 KK accumulator (from 128-thread kk_tiled_mma)
        kk_tiled_mma,
        kk_thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element for MMA operand
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) - used when needs_beta
        beta_pipe_idx: cutlass.Int32,
        tKKcMkk: cute.Tensor,  # coordinate mapping for KK fragment
    ):
        """Store tKKrKK → sKK_inv, Inverse, optionally reload+beta."""
        # stmatrix is sm_90. The thread-value map here comes from the MMA's C
        # layout, not from the atom, so swapping in an ordinary register-to-
        # shared copy leaves every element at the same address -- it just takes
        # one store per element instead of one instruction per 8x8 tile.
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.inverse_dtype)
        tiled_store = cute.make_tiled_copy_C(r2s_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_inv = cute.make_fragment_like(tKKrKK, self.inverse_dtype)
        tKKrKK_cv = thr_store.retile(tKKrKK_inv)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK[i])
        cute.copy(tiled_store, tKKrKK_cv, tKKsKK)

        cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
        CollectiveInverse(has_stmatrix=False).run(sKK_inv, NamedBarrier.KK_SYNC)

        if cutlass.const_expr(self.needs_beta or self.dtype != self.inverse_dtype):
            cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.inverse_dtype,
            )
            tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
            thr_load = tiled_load.get_slice(kk_thread_idx)
            tKKrKK_cpy = cute.make_fragment_like(tKKrKK_inv)
            tKKrKK_cvt = cute.make_fragment_like(tKKrKK_inv, self.dtype)
            tKKrKK_cv2 = thr_load.retile(tKKrKK_cpy)
            cute.copy(tiled_load, thr_load.partition_S(sKK_inv), tKKrKK_cv2)
            tKKcMkk_cv = thr_load.retile(tKKcMkk)

            for i in cutlass.range_constexpr(cute.size(tKKrKK_cpy)):
                if cutlass.const_expr(self.needs_beta):
                    _, t = tKKcMkk_cv[i]
                    tKKrKK_cvt[i] = self.dtype(
                        cutlass.Float32(tKKrKK_cpy[i])
                        * cutlass.Float32(sBeta[t, beta_pipe_idx])
                    )
                else:
                    tKKrKK_cvt[i] = self.dtype(tKKrKK_cpy[i])

            tKKsKK2 = thr_store.partition_D(sKK_opd)
            tKKrKK_cv3 = thr_store.retile(tKKrKK_cvt)
            cute.copy(tiled_store, tKKrKK_cv3, tKKsKK2)

    # ─── kk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        alpha_stage: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, t = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cute.math.exp2(
                    cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                    fastmath=True,
                )
        if cutlass.const_expr(self.needs_beta):
            beta_row = sBeta[None, beta_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, _ = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cutlass.Float32(beta_row[s])

    # ─── qk_or_kk_mask ────────────────────────────────────────────────────────

    @cute.jit
    def qk_or_kk_mask(
        self,
        frag: cute.Tensor,
        coord_tensor: cute.Tensor,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(frag)):
            s, t = coord_tensor[i]
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and (s < B and t < B)
            if not pred:
                frag[i] = cutlass.Float32(0.0)

    # ─── qk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def qk_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                s, t = tQKcMqk[i]
                tQKrQK[i] = (
                    tQKrQK[i]
                    * cute.math.exp2(
                        cutlass.Float32(alpha_cumlog[s])
                        - cutlass.Float32(alpha_cumlog[t]),
                        fastmath=True,
                    )
                    * scale
                )
        else:
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                tQKrQK[i] = tQKrQK[i] * scale

    # ─── qk_store ─────────────────────────────────────────────────────────────

    @cute.jit
    def qk_store(
        self,
        tQKrQK: cute.Tensor,
        sQK: cute.Tensor,
        qk_tiled_mma,
        qk_thread_idx: cutlass.Int32,
    ):
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.dtype)
        qk_tiled_copy = cute.make_tiled_copy_C(r2s_atom, qk_tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(qk_thread_idx)
        tQKsQK = qk_thr_copy.partition_D(sQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, self.dtype)
        tQKrQK_cvt_cv = qk_thr_copy.retile(tQKrQK_cvt)
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            tQKrQK_cvt[i] = self.dtype(tQKrQK[i])
        cute.copy(qk_tiled_copy, tQKrQK_cvt_cv, tQKsQK)

    # ─── o1_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def o1_epi(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cpscale = sAlpha[None, AlphaProcessor.CUMPROD_SCALE, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                _, tok_q = tOcO[i]
                tOrO[i] = cutlass.Float32(alpha_cpscale[tok_q]) * tOrO[i]
        else:
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                tOrO[i] = scale * tOrO[i]

    # ─── sk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def sk_epi(
        self,
        tSKrSK: cute.Tensor,
        tSKcSK: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cp = sAlpha[None, AlphaProcessor.CUMPROD, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tSKrSK)):
                _, tok_kv = tSKcSK[i]
                tSKrSK[i] = tSKrSK[i] * cutlass.Float32(alpha_cp[tok_kv])

    # ─── sk_load_v ────────────────────────────────────────────────────────────

    @cute.jit
    def sk_load_v(
        self,
        tSKrSK: cute.Tensor,
        sV_DS: cute.Tensor,
        sk_tiled_copy_C,
        sk_thr_copy_C,
        v_stage: cutlass.Int32,
    ) -> cute.Tensor:
        tSKrV = cute.make_fragment_like(tSKrSK, self.dtype)
        tSKrV_cv = sk_thr_copy_C.retile(tSKrV)
        tSKsV = sk_thr_copy_C.partition_S(sV_DS)
        cute.copy(sk_tiled_copy_C, tSKsV[None, None, None, v_stage], tSKrV_cv)
        return tSKrV

    # ─── kv_decay_v ───────────────────────────────────────────────────────────

    @cute.jit
    def kv_decay_v(
        self,
        tKVrV: cute.Tensor,
        tKVcV: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            block_log = cutlass.Float32(alpha_cumlog[B - cutlass.Int32(1)])
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                coeff = cute.math.exp2(
                    block_log - cutlass.Float32(alpha_cumlog[tok]), fastmath=True
                )
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        coeff = cutlass.Float32(0.0)
                tKVrV[i] = self.dtype(cutlass.Float32(tKVrV[i]) * coeff)
        else:
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        tKVrV[i] = self.dtype(0.0)

    # ─── o_store ──────────────────────────────────────────────────────────────

    @cute.jit
    def o_store(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sO: cute.Tensor,
    ):
        """Write the O accumulator to shared memory, one element at a time.

        The sm_90 path does this with stmatrix.trans: eight elements a thread
        per instruction, transposed on the way out. That instruction does not
        exist before sm_90, and no ordinary copy transposes, so the fragment is
        written by coordinate instead -- ``tOcO`` gives each register its
        (d, token) place, which is exactly how sO is laid out.

        The cost is instruction count, not bandwidth: the same bytes go to the
        same banks, but as one store per element rather than one per 8x8 tile.
        Indexing sO by logical coordinates keeps the swizzle applied, so the
        bank pattern is the one the layout was chosen for.

        No async-proxy fence either. The sm_90 path needs one because the
        reader of these bytes is the TMA engine, which sees shared memory
        through a different proxy than the stores that produced them. Nothing
        on this path does: the O tile is read back by ordinary loads in the
        store warp, so the barrier that warp already takes is what orders them,
        and fence.proxy.async.shared is an sm_90 instruction anyway.
        """
        for i in cutlass.range_constexpr(cute.size(tOrO)):
            d_i, tok = tOcO[i]
            sO[d_i, tok] = self.dtype(tOrO[i])

    # ─── cp.async load helpers ───────────────────────────────────────────────

    @cute.jit
    def _rep_slice(self, t: cute.Tensor, rep, d_is_mode1: cutlass.Constexpr):
        if cutlass.const_expr(d_is_mode1):
            return t[None, rep, 0]
        return t[None, 0, rep]

    @cute.jit
    def _aligned(self, t: cute.Tensor):
        """State the 16 B alignment the layout algebra cannot derive.

        Every offset folded into this pointer -- the lane's run of d, the token,
        the head, and the repeat above -- is a multiple of ``elems_per_lane``,
        so the address is always on a 16 B boundary. The token and head strides
        are runtime values, though, so the algebra falls back to element
        alignment and the atom is rejected for wanting 128 bits from a pointer
        it is told holds 16. Rounding up to 16 B is identity on a pointer
        already there, and it is what states the fact.

        It has to happen after the slice, not before: slicing folds another
        runtime stride into the address and drops the claim again.

        The premise is the caller's to keep; ``_check_load_alignment`` rejects
        inputs that break it before any of this runs.
        """
        return cute.make_tensor(t.iterator.align(_LOAD_ALIGN_BYTES), t.layout)

    @cute.jit
    def _copy_tile(
        self,
        gTile: cute.Tensor,
        sTile: cute.Tensor,
        rows: int,
        tid: cutlass.Int32,
        row_limit: cutlass.Int32,
        d_is_mode1: cutlass.Constexpr,
    ):
        """Issue one tile with cp.async, zero-filling rows past ``row_limit``.

        TMA clamped at the tensor bound on its own. Here the bound is a
        predicate, and a row the sequence does not own has to be zeroed rather
        than skipped: the math reads the whole tile, and the delta rule's
        masking assumes the padding contributes nothing.
        """
        if cutlass.const_expr(d_is_mode1):
            tv_shape, val_shape = self.q_load_tv_shape, self.q_load_val_shape
        else:
            tv_shape, val_shape = self.kv_load_tv_shape, self.kv_load_val_shape
        tv_layout = cute.make_layout(tv_shape[0], stride=tv_shape[1])
        val_layout = cute.make_layout(val_shape)
        load_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128
        )
        tiled_copy = cute.make_tiled_copy_tv(load_atom, tv_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tid)
        tSrc = thr_copy.partition_S(gTile)
        tDst = thr_copy.partition_D(sTile)
        # partition_S/D give (copy_atom_value, rest_m, rest_n): the lane's 16 B
        # run, then how many times the tiled copy has to repeat to cover the
        # tile. Only the row axis repeats here, so rest_n is one.
        rows_at_a_time = cutlass.Int32(LOAD_THREADS) // cutlass.Int32(
            self.D // self.elems_per_lane
        )
        lane_row = tid // cutlass.Int32(self.D // self.elems_per_lane)
        rep_mode = 1 if cutlass.const_expr(d_is_mode1) else 2
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[rep_mode])):
            row = cutlass.Int32(rep) * rows_at_a_time + lane_row
            # A row the sequence does not own is zeroed rather than skipped:
            # the math reads the whole tile and the masking assumes the padding
            # contributes nothing. TMA used to clamp this for free.
            if row < row_limit:
                cute.copy(
                    load_atom,
                    self._aligned(self._rep_slice(tSrc, rep, d_is_mode1)),
                    self._rep_slice(tDst, rep, d_is_mode1),
                )
            else:
                self._rep_slice(tDst, rep, d_is_mode1).fill(self.dtype(0.0))

    @cute.jit
    def load_qkv_cpasync(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
        # Rows of this tile the sequence actually owns. A full tile clamps to
        # the tile height, so the predicate costs nothing on the common path.
        rows_live = tok_end - blk_tok

        # K first, then Q, then V -- the order the math side waits in.
        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok), gK_full[None, None, k_head_idx]
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        k_pipeline.producer_acquire(k_producer_state)
        self._copy_tile(gK, sK, self.BLK_KV, tid, rows_live, False)
        cute.arch.cp_async_commit_group()
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)), gQ_full[None, None, q_head_idx]
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        q_pipeline.producer_acquire(q_producer_state)
        self._copy_tile(gQ, sQ, self.BLK_Q, tid, rows_live, True)
        cute.arch.cp_async_commit_group()
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        mV = cute.domain_offset(
            (cutlass.Int32(0), blk_tok), gV_full[None, None, v_head_idx]
        )
        gV = cute.zipped_divide(mV, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        v_pipeline.producer_acquire(v_producer_state)
        self._copy_tile(gV, sV, self.BLK_KV, tid, rows_live, False)
        cute.arch.cp_async_commit_group()
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    @cute.jit
    def issue_block_loads(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        alpha_pipeline,
        alpha_producer_state,
        beta_pipeline,
        beta_producer_state,
        blk: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        scale: cutlass.Float32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tid: cutlass.Int32,
        warp_idx: cutlass.Int32,
    ):
        """Fetch everything one block needs, with the whole block issuing it.

        The sm_90 kernel gives this to a warp group of its own and lets the
        math warps wait on an mbarrier, which this DSL will not emit for an
        sm_80 target -- and without one a thread can only wait on cp.async it
        issued itself. So the block does its own fetching and the math follows
        behind a barrier.

        Q, K and V go through cp.async and land when the consumer drains the
        group. Alpha and beta are scalar streams read with ordinary loads, so
        they are already in shared memory when this returns; the same barrier
        publishes them.
        """
        (
            q_producer_state,
            k_producer_state,
            v_producer_state,
        ) = self.load_qkv_cpasync(
            sQ_SD,
            sK_DS,
            sV_DS,
            gQ_full,
            gK_full,
            gV_full,
            q_pipeline,
            q_producer_state,
            k_pipeline,
            k_producer_state,
            v_pipeline,
            v_producer_state,
            blk,
            tok_start,
            tok_end,
            q_head_idx,
            k_head_idx,
            v_head_idx,
            tid,
        )

        # Alpha's scan reads and writes the same channel, so exactly one warp
        # may run it; a second would race the first between its load and its
        # store. Beta only writes, but it is kept to one warp for the same
        # reason there is nothing to gain from eight doing it.
        #
        # Neither branch contains a barrier, so restricting them does not make
        # the block's barrier participation uneven.
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.producer_acquire(alpha_producer_state)
            if warp_idx == cutlass.Int32(0):
                blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
                self.load_alpha(
                    sAlpha,
                    g_alpha,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                AlphaProcessor().run(
                    sAlpha[None, None, alpha_producer_state.index], scale
                )
            alpha_pipeline.producer_commit(alpha_producer_state)
            alpha_producer_state.advance()

        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.producer_acquire(beta_producer_state)
            if warp_idx == cutlass.Int32(1):
                blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
                self.load_beta(
                    sBeta,
                    g_beta,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    beta_producer_state.index,
                )
            beta_pipeline.producer_commit(beta_producer_state)
            beta_producer_state.advance()

        return (
            q_producer_state,
            k_producer_state,
            v_producer_state,
            alpha_producer_state,
            beta_producer_state,
        )

    # ─── load_alpha ───────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_alpha (scalar load).
    # Caller must sync before calling AlphaProcessor on the loaded data.

    @cute.jit
    def load_alpha(
        self,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sAlpha_k = sAlpha[None, None, alpha_stage]
        num_iters = self.BLK_Q // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = g_alpha[
                    tok * num_sab_heads + sab_head_idx
                ]
            else:
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = cutlass.Float32(1.0)

    # ─── load_beta ────────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_beta.

    @cute.jit
    def load_beta(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sBeta_k = sBeta[None, beta_stage]
        num_iters = self.BLK_KV // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sBeta_k[row] = g_beta[tok * num_sab_heads + sab_head_idx]
            else:
                sBeta_k[row] = cutlass.Float32(0.0)

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx].to(self.acc_dtype)

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i].to(gKV.element_type)

    @cute.jit
    def maybe_store_checkpoint(
        self,
        tKVrKV: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        kv_thr_mma,
        seq_idx: cutlass.Int32,
        o_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_checkpointing):
            if (
                block_end <= seq_len
                and block_end % checkpoint_every_n_tokens == cutlass.Int32(0)
            ):
                checkpoint_idx = (
                    cutlass.Int32(checkpoint_cu_starts[seq_idx])
                    + block_end // checkpoint_every_n_tokens
                    - cutlass.Int32(1)
                )
                checkpoint_layout = cute.make_ordered_layout(
                    (self.D, self.D, num_sab_heads, total_checkpoints),
                    order=(0, 1, 2, 3),
                )
                mCheckpoint = cute.make_tensor(
                    g_state_checkpoints.iterator, checkpoint_layout
                )
                gCheckpointKV = mCheckpoint[None, None, o_head_idx, checkpoint_idx]
                self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by Math WGs (tidx >= 128) for one block iteration.

    @cute.jit
    def compute_loop_body(
        self,
        # Smem tensors (staged; caller indexes the active stage)
        sQ_SD: cute.Tensor,  # (BlkQ, D, StagesQ)  – row-major atom, swizzled
        sK_SD: cute.Tensor,  # (BlkKV, D, StagesK) – same atom
        sK_DS: cute.Tensor,  # (D, BlkKV, StagesK) – K transposed
        sV_DS: cute.Tensor,  # (D, BlkKV, StagesV) – V transposed
        sQK: cute.Tensor,  # (BlkQ, BlkKV)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element
        sO: cute.Tensor,  # O output smem (staged)
        sAlpha: cute.Tensor,  # (BlkQ, AlphaProcessor.NUM_CHANNELS, StagesAlpha) or zero-shaped
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) or zero-shaped
        kv_tiled_mma,
        # Mainloop pipelines and active read states
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        # Compile-time flags
        is_first_block: bool,
        is_final_block: bool,
        # Valid token count for masking on final block
        B: cutlass.Int32,
        # Running KV state (D×D fp32, in registers across all blocks)
        tKVrKV: cute.Tensor,
        # Scale factor
        scale: cutlass.Float32,
        # WG role: MathWarpGroupRole.KK or MathWarpGroupRole.QK.
        wg_idx: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # Index among the math threads. On sm_90 they start at 128, behind the
        # load/store warp group; here there is no such group and they start at
        # zero, so the block index is already the one the MMA layouts want.
        # Subtracting 128 the way the sm_90 kernel does would hand the first
        # warp group negative coordinates, and the shared reads guarded by
        # `not is_first_block` would then address below the buffer.
        thread_idx = tidx
        kk_thread_idx = thread_idx % cutlass.Int32(128)
        qk_thread_idx = thread_idx % cutlass.Int32(128)

        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        tile_shape_qk = (blk_q, blk_kv, d)
        tile_shape_kk = tile_shape_qk
        tile_shape_o1 = (d, blk_q, d)
        tile_shape_o2 = (d, blk_q, blk_kv)
        tile_shape_sk = (d, blk_kv, d)
        tile_shape_newv = (d, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index

        mma_atom_4w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        mma_atom_8w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))

        # QK/KK: 4 warps × 16M = 64M  (1 warpgroup, 128 threads)
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_kk
        )

        # O1/O2/SK/NewV: 8 warps × 16M = 128M (both warpgroups, 256 threads)
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_newv
        )

        # ── Thread slices ─────────────────────────────────────────────────────
        qk_thr_mma = qk_tiled_mma.get_slice(qk_thread_idx)
        kk_thr_mma = kk_tiled_mma.get_slice(kk_thread_idx)
        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        # ── Copy atoms ────────────────────────────────────────────────────────
        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        # ── Active smem slices (extract 2D from staged tensors) ───────────────
        sQ_k = sQ_SD[None, None, q_stage]  # (BlkQ, D)
        sK_SD_k = sK_SD[None, None, k_stage]  # (BlkKV, D)
        sK_DS_k = sK_DS[None, None, k_stage]  # (D, BlkKV)

        # ── QK copies ─────────────────────────────────────────────────────────
        qk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, qk_tiled_mma)
        qk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, qk_tiled_mma)
        qk_thr_copy_A = qk_tiled_copy_A.get_slice(qk_thread_idx)
        qk_thr_copy_B = qk_tiled_copy_B.get_slice(qk_thread_idx)

        tQKrQ = qk_thr_mma.make_fragment_A(qk_thr_mma.partition_A(sQ_k))
        tQKrQ_cv = qk_thr_copy_A.retile(tQKrQ)
        tQKsQ = qk_thr_copy_A.partition_S(sQ_SD)
        tQKrK = qk_thr_mma.make_fragment_B(qk_thr_mma.partition_B(sK_SD_k))
        tQKrK_cv = qk_thr_copy_B.retile(tQKrK)
        tQKsK = qk_thr_copy_B.partition_S(sK_SD)

        # ── KK copies (same atom as QK) ───────────────────────────────────────
        kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, kk_tiled_mma)
        kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, kk_tiled_mma)
        kk_thr_copy_A = kk_tiled_copy_A.get_slice(kk_thread_idx)
        kk_thr_copy_B = kk_tiled_copy_B.get_slice(kk_thread_idx)

        tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD_k))
        tKKrA_cv = kk_thr_copy_A.retile(tKKrA)
        tKKsA = kk_thr_copy_A.partition_S(sK_SD)
        tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD_k))
        tKKrB_cv = kk_thr_copy_B.retile(tKKrB)
        tKKsB = kk_thr_copy_B.partition_S(sK_SD)

        # ── SK copies ─────────────────────────────────────────────────────────
        # SK B: K loaded from sK_SD (row-major BlkKV×D) with LDSM_N — matches C++ SK B-operand
        # SK C: V loaded from sV_DS (col-major D×BlkKV) with LDSM_T
        sk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, sk_tiled_mma)
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_B = sk_tiled_copy_B.get_slice(thread_idx)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)

        # Work around DSL make_fragment_B not accepting partition_shape_B output directly.
        tSKrK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_B(cute.slice_(tile_shape_sk, (0, None, None))),
            self.dtype,
        )
        tSKrK_cv = sk_thr_copy_B.retile(tSKrK)
        tSKsK = sk_thr_copy_B.partition_S(sK_SD)

        # ── NewV copies ───────────────────────────────────────────────────────
        newv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, newv_tiled_mma)
        newv_thr_copy_B = newv_tiled_copy_B.get_slice(thread_idx)
        tNewVrB = newv_thr_mma.make_fragment_B(newv_thr_mma.partition_B(sKK_opd))
        tNewVrB_cv = newv_thr_copy_B.retile(tNewVrB)
        tNewVsB = newv_thr_copy_B.partition_S(sKK_opd)

        # ── KV copies ─────────────────────────────────────────────────────────
        kv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_t4, kv_tiled_mma)
        kv_thr_copy_B = kv_tiled_copy_B.get_slice(thread_idx)
        tKVrK = kv_thr_mma.make_fragment_B(kv_thr_mma.partition_B(sK_DS_k))
        tKVrK_cv = kv_thr_copy_B.retile(tKVrK)
        tKVsK = kv_thr_copy_B.partition_S(sK_DS)

        # ── O1/O2 copies ──────────────────────────────────────────────────────
        o1_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o1_tiled_mma)
        o2_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o2_tiled_mma)
        o1_thr_copy_B = o1_tiled_copy_B.get_slice(thread_idx)
        o2_thr_copy_B = o2_tiled_copy_B.get_slice(thread_idx)

        # Direct partition_B(sQ_k) preserves the swizzled Q layout here and produces
        # a non-C++ B fragment shape; derive the fragment from TileShapeO1 instead.
        tOrQ = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_B(cute.slice_(tile_shape_o1, (0, None, None))),
            self.dtype,
        )
        tOrQ_cv = o1_thr_copy_B.retile(tOrQ)
        tOsQ = o1_thr_copy_B.partition_S(sQ_SD)
        tOrQK = o2_thr_mma.make_fragment_B(o2_thr_mma.partition_B(sQK))
        tOrQK_cv = o2_thr_copy_B.retile(tOrQK)
        tOsQK = o2_thr_copy_B.partition_S(sQK)

        # ── O store ───────────────────────────────────────────────────────────
        # The other two stores kept their thread-value map when the atom
        # changed, because they were writing the accumulator's own orientation.
        # This one was not: stmatrix.trans wrote the fragment out transposed,
        # and no ordinary copy does that. The accumulator and sO are both
        # (d, token), though, so the coordinates the MMA already hands out --
        # tOcO -- name the destination directly, and o_store walks them. See
        # o_store for what that costs.

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        cMkk = cMqk  # same shape (BlkKV == BlkQ == 64)
        tKKcMkk = kk_thr_mma.partition_C(cMkk)
        # Buffer reuse is ordered by the barrier that CollectiveStoreSm80
        # takes after writing O out, which sits between the last read here and
        # the next issue_block_loads. The consumer states below still advance,
        # because they name the stage; there is just no separate release
        # barrier for each of them.
        cO = cute.make_identity_tensor((d, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── KK GEMM (WG0 only) ────────────────────────────────────────────────
        # One wait for all five tensors, not one each. issue_block_loads has
        # already committed Q, K, V and, through ordinary stores, alpha and
        # beta, and prefetch is zero -- so this cp_async_wait_group(0) drains
        # every outstanding group and its barrier publishes all of it to the
        # block at once. A second wait would drain nothing and re-barrier.
        k_pipeline.consumer_wait(k_consumer_state)
        # Match the C++ reject-non-role-first shape; ptxas keeps BRA.U around
        # the role body instead of predicating the HMMA/LDSM/STSM sequence.
        if wg_idx != MathWarpGroupRole.KK:
            cute.arch.sync_warp()
        else:
            cute.copy(kk_tiled_copy_A, tKKsA[None, None, None, k_stage], tKKrA_cv)
            cute.copy(kk_tiled_copy_B, tKKsB[None, None, None, k_stage], tKKrB_cv)
            tKKrKK = cute.make_rmem_tensor(
                kk_thr_mma.partition_shape_C((blk_kv, blk_kv)), self.acc_dtype
            )
            tKKrKK.fill(self.acc_dtype(0.0))
            cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)
            self.kk_epi(tKKrKK, tKKcMkk, sAlpha, sBeta, alpha_stage, beta_stage)
            self.qk_or_kk_mask(tKKrKK, tKKcMkk, is_final_block, B)
            self._kk_store_and_inv(
                tKKrKK,
                kk_tiled_mma,
                kk_thread_idx,
                sKK_inv,
                sKK_opd,
                sBeta,
                beta_stage,
                tKKcMkk,
            )
        if cutlass.const_expr(self.needs_beta):
            beta_consumer_state.advance()

        # ── QK GEMM (WG1 only) ────────────────────────────────────────────────
        q_pipeline.consumer_wait(q_consumer_state)
        if wg_idx != MathWarpGroupRole.QK:
            cute.arch.sync_warp()
        else:
            cute.copy(qk_tiled_copy_A, tQKsQ[None, None, None, q_stage], tQKrQ_cv)
            cute.copy(qk_tiled_copy_B, tQKsK[None, None, None, k_stage], tQKrK_cv)
            tQKrQK = cute.make_rmem_tensor(
                qk_thr_mma.partition_shape_C((blk_q, blk_kv)), self.acc_dtype
            )
            tQKrQK.fill(self.acc_dtype(0.0))
            cute.gemm(qk_tiled_mma, tQKrQK, tQKrQ, tQKrK, tQKrQK)
            self.qk_epi(tQKrQK, tQKcMqk, sAlpha, alpha_stage, scale)
            self.qk_or_kk_mask(tQKrQK, tQKcMqk, is_final_block, B)
            self.qk_store(tQKrQK, sQK, qk_tiled_mma, qk_thread_idx)

        # ── O1: KV_state @ Q (both WGs, skip on first block) ─────────────────
        tOrO = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_C((d, blk_q)), self.acc_dtype
        )
        tOrO.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
            tOrKV = SM80.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
        q_consumer_state.advance()

        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        tSKrSK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            tSKrS = SM80.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            cute.copy(sk_tiled_copy_B, tSKsK[None, None, None, k_stage], tSKrK_cv)
            cute.gemm(sk_tiled_mma, tSKrSK, tSKrS, tSKrK, tSKrSK)

        # ── Load V from smem ──────────────────────────────────────────────────
        v_pipeline.consumer_wait(v_consumer_state)
        tSKrV = self.sk_load_v(tSKrSK, sV_DS, sk_tiled_copy_C, sk_thr_copy_C, v_stage)

        # sk_epi + V - SK  (SK=0 on first block, so V - SK = V)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_stage)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T  (ordered: WG0 first) ──────────────────────
        tNewVrA = SM80.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(
            newv_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        self._math_order_wait(wg_idx)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        cute.gemm(newv_tiled_mma, tNewVrC, tNewVrA, tNewVrB, tNewVrC)
        self._math_order_notify(wg_idx)
        v_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK  (ordered: WG0 first) ────────────────────────
        tOrNewV = SM80.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
        self._math_order_wait(wg_idx)
        cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
        cute.gemm(o2_tiled_mma, tOrO, tOrNewV, tOrQK, tOrO)
        self._math_order_notify(wg_idx)

        # ── O store to smem ───────────────────────────────────────────────────
        # No pipeline around this. With the store warp gone, the thread that
        # writes sO is the thread that will copy it out, and the caller's store
        # brackets that with the two barriers a handoff would have needed.
        self.o_store(tOrO, tOcO, sO[None, None, o_stage])

        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), AlphaProcessor.CUMPROD, alpha_stage]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)

        # KV += NewV @ K
        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tOrNewV, tKVrK, tKVrKV)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    # ─── Block entry point ───────────────────────────────────────────────────
    # One path for the whole block. The C++ kernel splits into warp roles here,
    # and that split needs an mbarrier this DSL will not emit for sm_80, so it
    # does not survive here -- see pipeline_sm80 for the measurements.

    @cute.jit
    def run_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gO_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        alpha_pipeline,
        beta_pipeline,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tok_end: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(self.D, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        packed_state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        state_ref_shape = (g_state.shape[0], g_state.shape[1], self.D, self.D)
        state_ref_layout = cute.make_layout(state_ref_shape, stride=g_state.stride)
        indexed_state_layout = cute.select(state_ref_layout, mode=[3, 2, 1, 0])
        state_idx = work_desc.seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices[work_desc.seq_idx])
            state_layout = indexed_state_layout
        else:
            state_layout = packed_state_layout
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        gStateKV = mState[None, None, o_head_idx, state_idx]
        if cutlass.const_expr(self.needs_init_state):
            init_state_ref_layout = cute.make_layout(
                state_ref_shape, stride=g_init_state.stride
            )
            indexed_init_state_layout = cute.select(
                init_state_ref_layout, mode=[3, 2, 1, 0]
            )
            if cutlass.const_expr(self.use_state_indices):
                init_state_layout = indexed_init_state_layout
            else:
                init_state_layout = packed_state_layout
            mInitState = cute.make_tensor(g_init_state.iterator, init_state_layout)
            gInitKV = mInitState[None, None, o_head_idx, state_idx]
            self.kv_load(tKVrKV, gInitKV, kv_thr_mma)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        if cutlass.const_expr(self.needs_init_state):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                beta_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sBeta,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_beta,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                beta_pipeline,
                beta_producer_state,
                cutlass.Int32(0),
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                cutlass.Int32(0),
                num_q_heads,
                num_v_heads,
                tidx,
            )
        else:
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                beta_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sBeta,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_beta,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                beta_pipeline,
                beta_producer_state,
                cutlass.Int32(0),
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                True,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                cutlass.Int32(0),
                num_q_heads,
                num_v_heads,
                tidx,
            )
        self.maybe_store_checkpoint(
            tKVrKV,
            g_state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            kv_thr_mma,
            work_desc.seq_idx,
            o_head_idx,
            num_sab_heads,
            total_checkpoints,
            cutlass.Int32(self.BLK_KV),
            work_desc.seq_len,
        )

        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                beta_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sBeta,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_beta,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                beta_pipeline,
                beta_producer_state,
                blk,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                blk,
                num_q_heads,
                num_v_heads,
                tidx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                work_desc.seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
            )

        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                beta_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sBeta,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_beta,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                beta_pipeline,
                beta_producer_state,
                last_blk,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                last_blk,
                num_q_heads,
                num_v_heads,
                tidx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                work_desc.seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
            )
        self.kv_store(tKVrKV, gStateKV, kv_thr_mma)

    # ─── Kernel entry point ───────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_o: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        grid_x: int,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )

        # One cp.async atom serves every load: the tile geometry lives in the
        # thread-value layout below rather than in a descriptor, so Q, K and V
        # differ only in how the loader partitions them.
        #
        # 128 bits a lane is the widest cp.async does, and it is what keeps a
        # tile down to one access per lane per row group. The value layout has
        # to run along the contiguous axis or the access is a gather and the
        # atom rejects it.
        # The atom itself is built inside the trace, not here: making it in
        # __init__ produces an IR value belonging to no kernel region, and the
        # tiled copy that consumes it is then rejected for using a value from
        # outside. Only the shape it implies is host state.
        elems_per_lane = 128 // self.dtype.width
        self.elems_per_lane = elems_per_lane
        # Q is (token, d) with d contiguous; K and V are (d, token) with d
        # contiguous. Both put d on the fast axis, so one tiled copy shape
        # covers all three: lanes split d, and successive thread rows walk the
        # other axis.
        # A lane's 16 B has to sit on whichever mode is contiguous, and the two
        # orientations disagree: Q is (token, d) so d is mode 1, while K and V
        # are (d, token) so d is mode 0. One layout pair each.
        lanes_along_d = self.D // elems_per_lane
        rows_at_a_time = LOAD_THREADS // lanes_along_d
        self.lanes_along_d = lanes_along_d
        self.load_rows_at_a_time = rows_at_a_time
        # Shapes only -- the layouts themselves are built inside the trace, for
        # the same reason the atom is.
        # Q: lanes split d within a row, thread rows walk tokens.
        self.q_load_tv_shape = ((rows_at_a_time, lanes_along_d), (lanes_along_d, 1))
        self.q_load_val_shape = (1, elems_per_lane)
        # K/V: the same split with the modes swapped, so the vector still runs
        # along d.
        self.kv_load_tv_shape = ((lanes_along_d, rows_at_a_time), (1, lanes_along_d))
        self.kv_load_val_shape = (elems_per_lane, 1)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        beta_storage_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))

        @cute.struct
        class SharedStorage:
            # The sm_90 path reserved two mbarriers a stage here, for the full
            # and empty phases. The sm_80 pipelines count async groups instead,
            # so that storage is gone and the stages have it back.
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[
                    self.inverse_dtype, cute.cosize(kk_storage_layout)
                ],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(alpha_storage_layout)
                ],
                16,
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_storage_layout)],
                16,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            g_alpha,
            g_beta,
            g_q,
            g_k,
            g_v,
            g_o,
            g_state,
            g_init_state,
            g_state_indices,
            g_state_checkpoints,
            checkpoint_cu_starts,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(LOAD_THREADS, 1, 1),
            max_number_threads=(LOAD_THREADS, 1, 1),
            stream=stream,
            # One. Asking for two costs 30% (1.751 -> 2.270 ms): the bound
            # forces ptxas to 128 registers a thread, half what this kernel
            # wants, and it reaches that by spilling. On the shapes where the
            # extra residency would matter the grid is also smaller than the
            # SM count -- it is num_seqs * num_sab_heads -- so there is no
            # second block to co-reside with in the first place.
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gO_full: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # The sm_90 path prefetched the four TMA descriptors here. cp.async has
        # no descriptor to warm, so the block goes straight to work.

        work_desc = self.get_next_work(
            cu_seqlens,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # With the load/store warp group gone, every thread is a math thread,
        # so its index within the math half is its index in the block.
        math_tidx = tidx
        wg_idx = tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        q_pipeline = PipelineCpAsyncSm80(self.q_stage)
        k_pipeline = PipelineCpAsyncSm80(self.k_stage)
        v_pipeline = PipelineCpAsyncSm80(self.v_stage)
        alpha_pipeline = PipelineCpAsyncSm80(self.alpha_beta_stage)
        beta_pipeline = PipelineCpAsyncSm80(self.alpha_beta_stage)
        # No mbarrier storage to fence: the stage pipelines are group
        # counters. One barrier still has to line the block up before the
        # first load.
        cute.arch.sync_threads()

        if work_desc.seq_len != cutlass.Int32(0):
            self.run_math_role(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                gQ_full,
                gK_full,
                gV_full,
                gO_full,
                g_alpha,
                g_beta,
                q_pipeline,
                k_pipeline,
                v_pipeline,
                alpha_pipeline,
                beta_pipeline,
                g_state,
                g_init_state,
                g_state_indices,
                g_state_checkpoints,
                checkpoint_cu_starts,
                work_desc,
                scale,
                wg_idx,
                math_tidx,
                tidx,
                warp_idx,
                tok_end,
                num_blocks,
                num_q_heads,
                num_v_heads,
                num_sab_heads,
                num_seqs,
                total_checkpoints,
                checkpoint_every_n_tokens,
            )


# ─── Public API ──────────────────────────────────────────────────────────────


@functools.cache
def _get_prefill_kernel(
    needs_alpha,
    needs_beta,
    needs_init_state,
    needs_checkpointing,
    kernel_dtype,
    initial_state_dtype,
    state_dtype,
    checkpoint_state_dtype,
    use_state_indices,
    cu_seqlens_dtype,
    state_indices_dtype,
    checkpoint_cu_starts_dtype,
    state_inner_strides,
    init_state_inner_strides,
):
    return _FullyFusedDeltaRuleSm80(
        needs_alpha,
        needs_beta,
        needs_init_state,
        needs_checkpointing,
        kernel_dtype,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        state_dtype=state_dtype_to_cutlass(state_dtype),
        checkpoint_state_dtype=state_dtype_to_cutlass(checkpoint_state_dtype),
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens_dtype,
        state_indices_dtype=state_indices_dtype,
        checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
        state_inner_strides=state_inner_strides,
        init_state_inner_strides=init_state_inner_strides,
    )


def delta_rule_prefill_dsl(
    o: torch.Tensor,  # (total_seqlen, num_o_heads, D) fp16/bf16, output
    state: torch.Tensor,  # (num_seqs, num_sab_heads, D, D) fp32, output
    q: torch.Tensor,  # (total_seqlen, num_q_heads, D)
    k: torch.Tensor,  # (total_seqlen, num_k_heads, D)
    v: torch.Tensor,  # (total_seqlen, num_v_heads, D)
    init_state: torch.Tensor | None,  # (num_seqs, num_sab_heads, D, D) fp32, optional
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,  # (num_seqs+1,) int64
    scale: float,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    state_indices: torch.Tensor | None = None,
):
    import cuda.bindings.driver as cuda_driver

    device = q.device
    D = q.shape[-1]

    num_seqs = cu_seqlens.shape[0] - 1
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)

    if not _FullyFusedDeltaRuleSm80.can_implement(
        num_q_heads, num_k_heads, num_v_heads, D, q.element_size()
    ):
        raise RuntimeError("can_implement failed")
    if D != 128:
        raise RuntimeError(f"DSL kernel only supports D=128, got {D}")

    needs_alpha = alpha is not None
    needs_beta = beta is not None
    needs_init_state = init_state is not None
    needs_checkpointing = checkpoint_every_n_tokens > 0
    use_state_indices = state_indices is not None
    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(q.dtype)
    if kernel_dtype is None:
        raise RuntimeError(f"DSL kernel only supports fp16/bf16 inputs, got {q.dtype}")

    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            f"q/k/v/o dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}, {o.dtype}"
        )
    if alpha is not None and alpha.dtype != torch.float32:
        raise RuntimeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if beta is not None and beta.dtype != torch.float32:
        raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
    if init_state is not None:
        state_dtype_to_cutlass(init_state.dtype)
    state_dtype_to_cutlass(state.dtype)
    if state_checkpoints is not None:
        state_dtype_to_cutlass(state_checkpoints.dtype)
    if not is_integer_dtype(cu_seqlens.dtype):
        raise RuntimeError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    expected_state_tail = (num_sab_heads, D, D)
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if (
            not use_state_indices and tensor.shape != (num_seqs, *expected_state_tail)
        ) or (use_state_indices and tuple(tensor.shape[1:]) != expected_state_tail):
            raise RuntimeError(
                f"{name} must have shape "
                f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_tail}, "
                f"got {tuple(tensor.shape)}"
            )
    if use_state_indices and (
        not is_integer_dtype(state_indices.dtype) or state_indices.shape != (num_seqs,)
    ):
        raise RuntimeError(
            f"state_indices must have shape {(num_seqs,)} and an integer dtype"
        )

    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("cu_seqlens", cu_seqlens),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (
        ("alpha", alpha),
        ("beta", beta),
        ("state_indices", state_indices),
        # state_checkpoints reaches the kernel as reshape(-1). On a tensor
        # that is not contiguous that returns a copy, so the kernel writes its
        # checkpoints into a temporary that is freed on return -- every
        # checkpoint lost, nothing raised. Measured: a pool sliced as
        # pool[::2] writes 0 of 4.
        ("state_checkpoints", state_checkpoints),
        # checkpoint_cu_starts is passed unreshaped, so that is not its
        # failure. Its problem is the compile cache: mark_layout_dynamic bakes
        # a unit stride whenever a dimension has one, and the cache key
        # carries dtypes and tile config but no strides -- so a first
        # contiguous call would bake stride 1 and a later strided one would
        # reuse that kernel and misread every offset.
        ("checkpoint_cu_starts", checkpoint_cu_starts),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if not use_state_indices and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    total_seqlen = q.shape[0]
    num_o_heads = o.shape[1]
    q_view = q.as_strided(
        (total_seqlen, D, num_q_heads),
        (num_q_heads * D, 1, D),
    )
    k_view = k.as_strided(
        (D, total_seqlen, num_k_heads),
        (1, num_k_heads * D, D),
    )
    v_view = v.as_strided(
        (D, total_seqlen, num_v_heads),
        (1, num_v_heads * D, D),
    )
    o_view = o.as_strided(
        (D, total_seqlen, num_o_heads),
        (1, num_o_heads * D, D),
    )
    _check_load_alignment(q=q_view, k=k_view, v=v_view, o=o_view)
    _check_state_dtype_supported(
        device,
        state=state,
        initial_state=init_state if needs_init_state else None,
        state_checkpoints=state_checkpoints if needs_checkpointing else None,
    )
    total_checkpoints = state_checkpoints.shape[0] if needs_checkpointing else 1

    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    delta_rule_kernel = _get_prefill_kernel(
        needs_alpha,
        needs_beta,
        needs_init_state,
        needs_checkpointing,
        kernel_dtype,
        initial_state_dtype=(init_state.dtype if needs_init_state else torch.float32),
        state_dtype=state.dtype,
        checkpoint_state_dtype=(
            state_checkpoints.dtype if needs_checkpointing else torch.float32
        ),
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens.dtype,
        state_indices_dtype=state_indices.dtype if use_state_indices else None,
        checkpoint_cu_starts_dtype=(
            checkpoint_cu_starts.dtype if needs_checkpointing else None
        ),
        state_inner_strides=(tuple(state.stride()[1:]) if use_state_indices else None),
        init_state_inner_strides=(
            tuple(init_state.stride()[1:])
            if use_state_indices and needs_init_state
            else None
        ),
    )

    compile_options = _sm80_compile_options(device)
    compiled_delta_rule_kernel = get_cached_compile(delta_rule_kernel, compile_options)
    if compiled_delta_rule_kernel is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(q_view, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(k_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(o_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            (
                from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_alpha
                else None
            ),
            (
                from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_beta
                else None
            ),
            from_dlpack(state, assumed_align=16).mark_layout_dynamic(),
            (
                from_dlpack(init_state, assumed_align=16).mark_layout_dynamic()
                if needs_init_state
                else None
            ),
            (
                from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
                if use_state_indices
                else None
            ),
            (
                from_dlpack(
                    state_checkpoints.reshape(-1), assumed_align=16
                ).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            (
                from_dlpack(checkpoint_cu_starts, assumed_align=8).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Float32(scale),
            cutlass.Int32(num_q_heads),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(num_seqs),
            cutlass.Int32(total_checkpoints),
            cutlass.Int32(checkpoint_every_n_tokens),
            num_seqs * num_sab_heads,
            stream,
        )
        compiled_delta_rule_kernel = cached_compile(
            delta_rule_kernel,
            *kernel_args,
            compile_options=compile_options,
        )
    compiled_delta_rule_kernel(
        q_view,
        k_view,
        v_view,
        o_view,
        alpha.reshape(-1) if needs_alpha else None,
        beta.reshape(-1) if needs_beta else None,
        state,
        init_state if needs_init_state else None,
        state_indices if use_state_indices else None,
        state_checkpoints.reshape(-1) if needs_checkpointing else None,
        checkpoint_cu_starts if needs_checkpointing else None,
        cu_seqlens,
        scale,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        num_seqs,
        total_checkpoints,
        checkpoint_every_n_tokens,
        num_seqs * num_sab_heads,
        stream,
    )
