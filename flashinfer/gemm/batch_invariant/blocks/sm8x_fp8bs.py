"""The SM8x body of the block-scale fp8 class ``gemm.e4m3.kblock128`` (SM80 / SM86 / SM87 / SM89).

The block partials ``P_kb`` are the per-tensor fp8 class's SM8x fold of each 128-wide K block from
+0 (the SM80 ``mma.sync`` k16 fold of `flashinfer.gemm.sm8x_fp8_dense`, unit scales, fp32 out: one
batched GEMM over the blocks, the weight read in place), and `Promote` folds them in block order,
``acc = fma.rn(mul.rn(s_a[m, kb], s_b[n // 128, kb]), P_kb, acc)`` from +0, then the single
``cvt.rn``. The rows are independent, so the partials are formed per row slab (a placement). The
callable takes the SM100 body's operands (`blocks.compile_fp8bs`); its knobs, persistent grid and
scale staging are placements of that body and are not read.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from ....batch_invariant import jit
from ....batch_invariant.fp_interface.cute import CuteF32
from ..spec import BLOCK_K, SCALE_BLOCK_N
from .operands import OUT_DT

THREADS = 256
MAX_GRID_Y = 65535  # the hardware's grid.y limit; rows beyond it are strided over
#: The fp32 block partials of one row slab (bytes); at least 64 rows per slab.
SLAB_BYTES = 1 << 28


class Promote:
    """``out[m, n] = cvt(acc_KB)``, ``acc_0 = +0``, ``acc_{kb+1} = fma.rn(mul.rn(sa[m, kb], sb[n // 128, kb]),
    P[m, n, kb], acc_kb)``. One thread per output column; block ``by`` handles rows ``by, by + rows, ...``."""

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mSa: cute.Tensor,
        mSb: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        N: Int32,
        KB: Int32,
        rows: Int32,
        stream,
    ):
        self.kernel(mP, mSa, mSb, mY, M, N, KB, rows).launch(
            grid=[cute.ceil_div(N, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mSa: cute.Tensor,
        mSb: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        N: Int32,
        KB: Int32,
        rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        n = bx * THREADS + tidx
        if n < N:
            nb = n // SCALE_BLOCK_N
            for m in cutlass.range(by, M, rows, unroll=1):
                acc = CuteF32.const(0.0)  # acc_0 = +0
                for kb in cutlass.range(0, KB, unroll=1):  # blocks ascending
                    scale = CuteF32.mul(mSa[m, kb], mSb[nb, kb])
                    acc = CuteF32.fma(scale, mP[m, n, kb], acc)
                mY[m, n] = acc.to(mY.element_type)  # the chain's single cvt.rn


@functools.cache
def get_promote(module: str, out_fmt: str):
    """`Promote` for one output format in the cache module ``module``; M, N and the block count are runtime arguments,
    the partials and the scales take any strides."""
    out_dt = OUT_DT[out_fmt]

    def compile_fn():
        sym_m, sym_n = cute.sym_int(32), cute.sym_int(32)
        sym_kb, sym_nb = cute.sym_int(32), cute.sym_int(32)
        p_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_n, sym_kb),
            (cute.sym_int64(), 1, cute.sym_int64()),
            assumed_align=4,
        )
        sa_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        sb_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_nb, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        y_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n),
            (cute.sym_int64(), 1),
            assumed_align=out_dt.width // 8,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Promote(),
            p_fake,
            sa_fake,
            sb_fake,
            y_fake,
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        module, jit.sanitize_symbol_name(f"sm8x_promote_{out_fmt}"), compile_fn
    )


def fp8bs_callable(module: str, out_fmt: str):
    """``run(a3, b3, sfa, sfb, c3)`` with the operands of `blocks.compile_fp8bs.get_compiled_fp8bs`: A ``(M, K, 1)``,
    B ``(N, K, 1)``, the scale views ``(M, K / 128)`` and ``(ceil(N / 128), K / 128)``, C ``(M, N, 1)``."""
    from ...sm8x_fp4_dense import _ones
    from ...sm8x_fp8_dense import _run_fp8, row_tile

    promote = get_promote(module, out_fmt)

    def run(a3, b3, sfa, sfb, c3):
        M, K, N = int(a3.shape[0]), int(a3.shape[1]), int(b3.shape[0])
        if M == 0:
            return
        KB, one = K // BLOCK_K, _ones(a3.device)
        a, b, c = a3[:, :, 0], b3[:, :, 0], c3[:, :, 0]
        sa, sb = a.stride(0), b.stride(0)
        bk = b.as_strided((KB, BLOCK_K, N), (BLOCK_K, 1, sb))  # block kb: (128, N)
        rows = SLAB_BYTES // (KB * N * 4) // 64 * 64
        step = M if rows >= M else max(64, rows)
        for m0 in range(0, M, step):
            mc = min(step, M - m0)
            ak = a[m0:].as_strided((KB, mc, BLOCK_K), (BLOCK_K, sa, 1))
            P = torch.empty((KB, mc, N), dtype=torch.float32, device=a3.device)
            _run_fp8(ak, bk, one, one, P, row_tile(mc), False)
            y = c[m0 : m0 + mc]
            P3, sfa_m = P.permute(1, 2, 0), sfa[m0 : m0 + mc]
            promote(P3, sfa_m, sfb, y, mc, N, KB, min(mc, MAX_GRID_Y))

    return run
