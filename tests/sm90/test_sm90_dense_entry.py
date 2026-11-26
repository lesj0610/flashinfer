"""SM90 (wgmma) dense FP4 entry points: ``mm_bf16_fp4``, ``mm_fp4`` and the cake per-token
backend, on the operands in each entry point's own SM100+ format.

Every case (1) builds its operands with the torch quantizer below (no FlashInfer quantizer,
so the reference does not share code with the path under test), (2) takes the expected
value from a torch matmul on the dequantized operands, and (3) asserts through the CUDA
profiler that the kernel that ran is the SM90 wgmma kernel (``fp4_sm90_gemm``) and not a
library kernel. A case that runs another kernel, or none, fails.

Cases follow the rows of the dense read tables: D1-D7 (A path, tactic shapes, epilogue),
F1-F13 (entry points by weight format) and E1/E2 (errors), plus the ``mm_fp4`` auto rows
2-S1..2-S12, which no other test runs on any architecture.
"""

import pytest
import torch

from flashinfer.utils import BackendSupportedError, get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) != (9, 0),
    reason="SM90 dense FP4 entry points",
)

KERNEL_SYMBOL = "fp4_sm90_gemm"
FOREIGN_SYMBOLS = ("cutlass", "cudnn", "cute", "sm100", "sm120", "gemv")
GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# torch quantizers and scale layouts (independent of FlashInfer's)
# ---------------------------------------------------------------------------


def _quantize(x: torch.Tensor, nvfp4: bool):
    """Block-scaled FP4: (packed uint8 [R, K/2], scale bytes [R, K/B], dequantized fp32)."""
    r, k = x.shape
    block = 16 if nvfp4 else 32
    xb = x.float().reshape(r, k // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if nvfp4:
        scale = (amax / 6.0).to(torch.float8_e4m3fn)
        sf = scale.view(torch.uint8).reshape(r, k // block)
        scale_f = scale.float()
    else:
        e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
        sf = (e + 127).to(torch.uint8).reshape(r, k // block)
        scale_f = torch.exp2(e)
    scale_f = scale_f.clamp_min(1e-30)
    grid = torch.tensor(GRID, device=x.device)
    mag = (xb / scale_f).abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(-1)
    neg = (xb < 0) & (idx != 0)
    code = (idx + 8 * neg.to(idx.dtype)).reshape(r, k).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    deq = grid[idx] * torch.where(neg, -1.0, 1.0) * scale_f
    return packed.contiguous(), sf.contiguous(), deq.reshape(r, k)


def _swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """[R, C] scale bytes to the 128x4 tiled layout, flat, rows/cols padded to 128/4."""
    r, c = sf.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    ri = torch.arange(rp, device=sf.device)[:, None]
    ci = torch.arange(cp, device=sf.device)[None, :]
    off = (
        (ri // 128) * (cp // 4) * 512
        + (ci // 4) * 512
        + (ri % 32) * 16
        + ((ri % 128) // 32) * 4
        + ci % 4
    )
    out = torch.zeros(rp * cp, dtype=torch.uint8, device=sf.device)
    padded = torch.zeros(rp, cp, dtype=torch.uint8, device=sf.device)
    padded[:r, :c] = sf
    out[off.reshape(-1)] = padded.reshape(-1)
    return out


def _swizzle_8x4(sf: torch.Tensor) -> torch.Tensor:
    """[R, C] to the 8x4 tiled layout (8 rows x 4 columns per 32-byte tile), flat."""
    r, c = sf.shape
    rp, cp = -(-r // 8) * 8, -(-c // 4) * 4
    ri = torch.arange(rp, device=sf.device)[:, None]
    ci = torch.arange(cp, device=sf.device)[None, :]
    off = (ri // 8) * (cp // 4) * 32 + (ci // 4) * 32 + (ri % 8) * 4 + ci % 4
    out = torch.zeros(rp * cp, dtype=torch.uint8, device=sf.device)
    padded = torch.zeros(rp, cp, dtype=torch.uint8, device=sf.device)
    padded[:r, :c] = sf
    out[off.reshape(-1)] = padded.reshape(-1)
    return out


def _trtllm_shuffle(t: torch.Tensor) -> torch.Tensor:
    """Row i of every 32-row block moves to row (i % 4) * 8 + i // 4 (the trtllm dense
    ``srcToDstBlk32RowMap``)."""
    rows = t.shape[0]
    assert rows % 32 == 0
    i = torch.arange(rows, device=t.device)
    dst = (i // 32) * 32 + (i % 4) * 8 + (i % 32) // 4
    out = torch.empty_like(t)
    out[dst] = t
    return out


class Operand:
    """One quantized matrix in the layouts an entry point can take."""

    def __init__(self, x: torch.Tensor, nvfp4: bool = True):
        self.x = x
        self.nvfp4 = nvfp4
        self.codes, self.sf, self.deq = _quantize(x, nvfp4)
        self.rows, self.k = x.shape

    def swz(self):
        return _swizzle_128x4(self.sf).reshape(-1, -(-self.sf.shape[1] // 4) * 4)

    def swz8(self):
        return _swizzle_8x4(self.sf).reshape(-1, -(-self.sf.shape[1] // 4) * 4)

    def shuffled(self):
        """trtllm weight: shuffled codes and shuffled scales, scales then 128x4-tiled."""
        codes = _trtllm_shuffle(self.codes)
        sf = _trtllm_shuffle(self.sf)
        return codes, _swizzle_128x4(sf).reshape(-1, -(-sf.shape[1] // 4) * 4)


def _rand(m, k, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), device="cuda", dtype=torch.float32, generator=g).to(
        dtype
    )


def _check(out: torch.Tensor, ref: torch.Tensor, what: str):
    ref = ref.float()
    err = (out.float() - ref).abs().max().item()
    assert err <= 2e-2 * ref.abs().max().item() + 1e-6, f"{what}: max abs err {err}"


# ---------------------------------------------------------------------------
# which kernel ran
# ---------------------------------------------------------------------------


def _run_profiled(fn):
    """Run ``fn`` (after one warm-up call) under the CUDA profiler; return its result and
    the names of the kernels launched."""
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]
    return result, names


def _assert_sm90_kernel(names, what):
    assert any(KERNEL_SYMBOL in n for n in names), (
        f"{what}: no {KERNEL_SYMBOL}: {names}"
    )
    # Own kernels first: their argument lists (CUtensorMap_st) contain "cute".
    others = [n for n in names if "fp4_sm90" not in n and "fp4_dense_sm90" not in n]
    gemms = [n for n in others if any(s in n.lower() for s in FOREIGN_SYMBOLS)]
    assert not gemms, f"{what}: a non-SM90 kernel ran: {gemms}"


def _run_checked(fn, ref, what):
    out, names = _run_profiled(fn)
    _assert_sm90_kernel(names, what)
    _check(out, ref, what)
    return out


# ---------------------------------------------------------------------------
# D1-D7 and F1-F5: the kernel through sm90_fp4_dense, by weight format
# ---------------------------------------------------------------------------

from flashinfer.fused_moe.sm90.spec import AFmt  # noqa: E402
from flashinfer.gemm import sm90_fp4_dense as dense  # noqa: E402


def _view(fmt, w: Operand):
    if fmt == "nvfp4_swz128x4":
        return dense.sm90_dense_weight_view(
            "t", AFmt.NVFP4_SWZ128X4, w.codes, w.swz(), w.rows, w.k
        )
    if fmt == "nvfp4_linear_sf":
        return dense.sm90_dense_weight_view(
            "t", AFmt.NVFP4_LINEAR_SF, w.codes, w.sf, w.rows, w.k
        )
    if fmt == "nvfp4_trtllm_shuffled":
        codes, sf = w.shuffled()
        return dense.sm90_dense_weight_view(
            "t", AFmt.NVFP4_TRTLLM_SHUFFLED, codes, sf, w.rows, w.k
        )
    if fmt == "mxfp4_swz128x4":
        return dense.sm90_dense_weight_view(
            "t", AFmt.MXFP4_SWZ128X4, w.codes, w.swz(), w.rows, w.k
        )
    raise AssertionError(fmt)


@pytest.mark.parametrize("a_path", [0, 1])
@pytest.mark.parametrize("act", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "wfmt", ["nvfp4_swz128x4", "nvfp4_linear_sf", "nvfp4_trtllm_shuffled"]
)
def test_every_tactic_w4a16(wfmt, act, a_path):
    """D1, D7, F1: every tactic of the table, ragged N (rows) and M, 16-bit activations."""
    n_rows, k = 416, 512
    w = Operand(_rand(n_rows, k, 1))
    view = _view(wfmt, w)
    f16 = act == torch.float16
    tactics = dense._module_tactics(f16, view.a_fmt, a_path)
    assert len(tactics) == 11
    for tactic in tactics:
        for m in (1, tactic[1] + 3):
            a = _rand(m, k, 2 + m, act)
            ref = a.float() @ w.deq.T
            _run_checked(
                lambda: dense.sm90_fp4_dense(
                    a, None, view, None, act, tactic=tactic, a_path=a_path
                ),
                ref,
                f"{wfmt} {act} a_path={a_path} tactic={tactic} m={m}",
            )


@pytest.mark.parametrize("alpha_kind", ["none", "scalar", "per_token"])
def test_alpha_forms(alpha_kind):
    """D2, F7: alpha None / scalar / one per row, applied in the epilogue."""
    m, n_rows, k = 37, 256, 512
    w = Operand(_rand(n_rows, k, 3))
    a = _rand(m, k, 4)
    ref = a.float() @ w.deq.T
    if alpha_kind == "scalar":
        alpha = torch.tensor([0.37], device="cuda")
        ref = ref * 0.37
    elif alpha_kind == "per_token":
        alpha = 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m
        ref = ref * alpha[:, None]
    else:
        alpha = None
    view = _view("nvfp4_swz128x4", w)
    _run_checked(
        lambda: dense.sm90_fp4_dense(a, None, view, alpha, torch.bfloat16),
        ref,
        f"alpha {alpha_kind}",
    )


@pytest.mark.parametrize("layout", ["128x4", "8x4"])
@pytest.mark.parametrize("wfmt", ["nvfp4_swz128x4", "nvfp4_trtllm_shuffled"])
def test_w4a4(wfmt, layout):
    """D3, F2, F5: FP4 activations (prep restores them), 128x4 and 8x4 scale layouts."""
    m, n_rows, k = 40, 256, 512
    w = Operand(_rand(n_rows, k, 5))
    x = Operand(_rand(m, k, 6))
    sfa = x.swz() if layout == "128x4" else x.swz8()
    view = _view(wfmt, w)
    alpha = torch.tensor([0.5], device="cuda")
    ref = (x.deq @ w.deq.T) * 0.5
    _run_checked(
        lambda: dense.sm90_fp4_dense(
            x.codes, sfa, view, alpha, torch.bfloat16, a_layout=layout
        ),
        ref,
        f"w4a4 {wfmt} {layout}",
    )


def test_mxfp4_weight_w4a16():
    """F3, F6: MXFP4 weights with BF16 activations."""
    m, n_rows, k = 24, 256, 512
    w = Operand(_rand(n_rows, k, 7), nvfp4=False)
    a = _rand(m, k, 8)
    view = _view("mxfp4_swz128x4", w)
    _run_checked(
        lambda: dense.sm90_fp4_dense(a, None, view, None, torch.bfloat16),
        a.float() @ w.deq.T,
        "mxfp4 w4a16",
    )


def test_output_is_row_major_and_preallocated():
    """D4: out [M, N] row-major, written in place into ``out``."""
    m, n_rows, k = 9, 192, 256
    w = Operand(_rand(n_rows, k, 9))
    a = _rand(m, k, 10)
    out = torch.full((m, n_rows), float("nan"), device="cuda", dtype=torch.bfloat16)
    view = _view("nvfp4_swz128x4", w)
    res = _run_checked(
        lambda: dense.sm90_fp4_dense(a, None, view, None, torch.bfloat16, out=out),
        a.float() @ w.deq.T,
        "out=",
    )
    assert res.data_ptr() == out.data_ptr()


# ---------------------------------------------------------------------------
# F9-F13: the public entry points
# ---------------------------------------------------------------------------

from flashinfer import mm_fp4  # noqa: E402
from flashinfer.gemm import gemm_base  # noqa: E402
from flashinfer.gemm.gemm_bf16_fp4 import mm_bf16_fp4, prepare_bf16_fp4_weights  # noqa: E402


def _w4a16_canonical(n_rows, k, seed):
    """Weights as ``prepare_bf16_fp4_weights`` expects them: uint8 [N, K/2] codes, swizzled
    128x4 scale bytes, and alpha (a 1-element fp32 tensor)."""
    w = Operand(_rand(n_rows, k, seed))
    alpha = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    return w, w.codes, w.swz().reshape(-1), alpha


@pytest.mark.parametrize(
    "backend,out_dtype",
    [
        ("cudnn", torch.bfloat16),
        ("cudnn", torch.float16),
        ("cute-dsl", torch.bfloat16),
        ("cute-dsl-native", torch.bfloat16),
        ("blackwell-native", torch.bfloat16),
        ("blackwell-tiled", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("m", [1, 16, 130])
def test_mm_bf16_fp4_backends(backend, out_dtype, m):
    """F9, F10, F4: each mm_bf16_fp4 backend name, run through its own prepare."""
    n_rows, k = 512, 1024
    w, codes, sf, alpha = _w4a16_canonical(n_rows, k, 11)
    a = _rand(m, k, 12, torch.bfloat16)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend=backend)
    ref = (a.float() @ w.deq.T) * 0.75
    _run_checked(
        lambda: mm_bf16_fp4(
            a, b_p, sf_p, alpha_p, backend=backend, out_dtype=out_dtype
        ),
        ref,
        f"mm_bf16_fp4 {backend} {out_dtype} m={m}",
    )


def test_mm_bf16_fp4_rejects_fp16_activations():
    """The shared requirement of mm_bf16_fp4 takes BF16 activations only; unchanged on SM90."""
    w, codes, sf, alpha = _w4a16_canonical(256, 512, 13)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend="cudnn")
    a = _rand(4, 512, 14, torch.float16)
    with pytest.raises((ValueError, TypeError, BackendSupportedError)):
        mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend="cudnn")


def _fp4_pair(m, n_rows, k, seed, nvfp4=True):
    a = Operand(_rand(m, k, seed), nvfp4)
    w = Operand(_rand(n_rows, k, seed + 100), nvfp4)
    return a, w


def _mm_fp4_call(
    a: Operand, w: Operand, backend, out_dtype, alpha, layout="128x4", nvfp4=True
):
    b, sfb = (w.codes, w.swz())
    if backend == "trtllm":
        b, sfb = w.shuffled()
    sfa = a.swz() if layout == "128x4" else a.swz8()
    out = torch.empty((a.rows, w.rows), device="cuda", dtype=out_dtype)
    return mm_fp4(
        a.codes,
        b.T,
        sfa,
        sfb.T,
        alpha,
        out_dtype,
        out,
        block_size=16 if nvfp4 else 32,
        use_8x4_sf_layout=layout == "8x4",
        backend=backend,
        use_nvfp4=nvfp4,
    )


@pytest.mark.parametrize("m", [1, 8, 64])
@pytest.mark.parametrize(
    "backend,out_dtype",
    [
        ("cudnn", torch.bfloat16),
        ("cudnn", torch.float16),
        ("cutlass", torch.bfloat16),
        ("cutlass", torch.float16),
        ("trtllm", torch.bfloat16),
        ("cute-dsl", torch.bfloat16),
        ("cute-dsl", torch.float16),
        ("cutedsl_low_latency", torch.bfloat16),
        ("auto", torch.bfloat16),
        ("auto", torch.float16),
    ],
)
def test_mm_fp4_nvfp4_backends(backend, out_dtype, m):
    """F12, F13 and 2-S2/2-S3: NVFP4 scalar alpha for every backend name and for auto."""
    a, w = _fp4_pair(m, 512, 512, 15)
    alpha = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    if backend == "cutedsl_low_latency" and m > 8:
        # The SM100+ requirement for this name: M <= 8.
        with pytest.raises(ValueError, match="M <= 8"):
            _mm_fp4_call(a, w, backend, out_dtype, alpha)
        return
    ref = (a.deq @ w.deq.T) * 0.5
    _run_checked(
        lambda: _mm_fp4_call(a, w, backend, out_dtype, alpha),
        ref,
        f"mm_fp4 {backend} {out_dtype} m={m}",
    )


def test_mm_fp4_trtllm_8x4():
    """F5: trtllm takes the 8x4 activation scale layout."""
    a, w = _fp4_pair(24, 256, 512, 16)
    alpha = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    _run_checked(
        lambda: _mm_fp4_call(a, w, "trtllm", torch.bfloat16, alpha, layout="8x4"),
        (a.deq @ w.deq.T) * 0.5,
        "mm_fp4 trtllm 8x4",
    )


@pytest.mark.parametrize(
    "backend", ["cudnn", "cute-dsl", "cutedsl_low_latency", "auto"]
)
@pytest.mark.parametrize("m", [1, 8])
def test_mm_fp4_mxfp4(backend, m):
    """F3, F6, 2-S4: MXFP4 (block 32, e8m0 scales), no alpha; the names that accept it."""
    a, w = _fp4_pair(m, 512, 1024, 17, nvfp4=False)
    _run_checked(
        lambda: _mm_fp4_call(a, w, backend, torch.bfloat16, None, nvfp4=False),
        a.deq @ w.deq.T,
        f"mm_fp4 mxfp4 {backend} m={m}",
    )


def test_mm_fp4_mxfp4_cutlass_rejected():
    """2-S4: cutlass has no MXFP4 (ValueError in its requirement), on SM90 as on SM100."""
    a, w = _fp4_pair(8, 256, 1024, 18, nvfp4=False)
    with pytest.raises(ValueError):
        _mm_fp4_call(a, w, "cutlass", torch.bfloat16, None, nvfp4=False)


# ---- auto rows 2-S1..2-S12 -------------------------------------------------


def test_auto_s1_batch_invariant_unsupported():
    """2-S1: batch_invariant is M6 (not in M3): a clear error, not a silent other path."""
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    a, w = _fp4_pair(8, 256, 512, 19)
    out = torch.empty((8, 256), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(BatchInvariantUnsupportedError):
        mm_fp4(
            a.codes,
            w.codes.T,
            a.swz(),
            w.swz().T,
            None,
            torch.bfloat16,
            out,
            block_size=16,
            backend="auto",
            use_nvfp4=True,
            batch_invariant=True,
        )


def test_auto_s5_no_cudnn_large_m_mxfp4():
    """2-S5: MXFP4, M > 8 and no cuDNN: no candidate, an error naming 'auto'."""
    if gemm_base.CUDNN_AVAILABLE:
        pytest.skip("row needs a cuDNN-less environment (cuDNN is present here)")
    a, w = _fp4_pair(16, 256, 1024, 20, nvfp4=False)
    with pytest.raises(BackendSupportedError):
        _mm_fp4_call(a, w, "auto", torch.bfloat16, None, nvfp4=False)


@pytest.mark.parametrize("nvfp4", [True, False])
def test_auto_s6_8x4_has_no_candidate(nvfp4):
    """2-S6: the 8x4 layout with auto: only trtllm accepts it and it is not an auto
    candidate, so the call fails with the no-candidate error."""
    a, w = _fp4_pair(8, 256, 1024, 21, nvfp4=nvfp4)
    alpha = torch.tensor([0.5], device="cuda") if nvfp4 else None
    with pytest.raises((BackendSupportedError, ValueError)):
        _mm_fp4_call(a, w, "auto", torch.bfloat16, alpha, layout="8x4", nvfp4=nvfp4)


def _per_token_alpha(m):
    return 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 17, 130])
def test_per_token_alpha_rows(backend, out_dtype, m):
    """2-S7, F7: NVFP4 per-token alpha (N % 8 == 0), cute-dsl and auto; each row scaled by
    its own alpha."""
    a, w = _fp4_pair(m, 512, 512, 22)
    alpha = _per_token_alpha(m)
    ref = (a.deq @ w.deq.T) * alpha[:, None]
    _run_checked(
        lambda: _mm_fp4_call(a, w, backend, out_dtype, alpha),
        ref,
        f"per-token {backend} m={m}",
    )


def test_per_token_alpha_mxfp4():
    """2-S9: MXFP4 per-token alpha through cute-dsl (the prediction of the SM100 rows)."""
    m = 16
    a, w = _fp4_pair(m, 256, 1024, 23, nvfp4=False)
    alpha = _per_token_alpha(m)
    _run_checked(
        lambda: _mm_fp4_call(a, w, "cute-dsl", torch.bfloat16, alpha, nvfp4=False),
        (a.deq @ w.deq.T) * alpha[:, None],
        "per-token mxfp4 cute-dsl",
    )


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
def test_auto_s10_per_token_misaligned_n(backend):
    """2-S10: per-token alpha with N % 8 != 0 has no backend: an error, before any kernel."""
    m = 16
    a, w = _fp4_pair(m, 130, 512, 24)
    with pytest.raises((BackendSupportedError, ValueError)):
        _mm_fp4_call(a, w, backend, torch.bfloat16, _per_token_alpha(m))


def test_auto_s12_float32_output_rejected():
    """2-S12: the shared argument check still rejects an unsupported output dtype."""
    a, w = _fp4_pair(8, 256, 512, 25)
    alpha = torch.tensor([0.5], device="cuda")
    with pytest.raises((ValueError, TypeError, BackendSupportedError)):
        _mm_fp4_call(a, w, "cute-dsl", torch.float32, alpha)


def test_e2_cute_dsl_misaligned_n_raises():
    """E2: cute-dsl NVFP4 with N % 8 != 0 stays an error on SM90 (the SM12x tuple needs
    N % 64 == 0, so the union of the SM10x/SM12x tuples does not widen it)."""
    a, w = _fp4_pair(16, 130, 128, 26)
    alpha = torch.tensor([0.5], device="cuda")
    with pytest.raises((BackendSupportedError, ValueError)):
        _mm_fp4_call(a, w, "cute-dsl", torch.bfloat16, alpha)


# ---------------------------------------------------------------------------
# F11: cake per-token backend (experimental, env-gated)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 33])
def test_cake_per_token(monkeypatch, m):
    """F11: backend='cake' with a per-token alpha runs the SM90 kernel and rejects N < 32."""
    monkeypatch.delenv("FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES", raising=False)
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    a, w = _fp4_pair(m, 256, 512, 27)
    alpha = _per_token_alpha(m)
    _run_checked(
        lambda: _mm_fp4_call(a, w, "cake", torch.bfloat16, alpha),
        (a.deq @ w.deq.T) * alpha[:, None],
        f"cake per-token m={m}",
    )
    small, small_w = _fp4_pair(m, 16, 512, 28)
    with pytest.raises((ValueError, BackendSupportedError)):
        _mm_fp4_call(small, small_w, "cake", torch.bfloat16, alpha)
