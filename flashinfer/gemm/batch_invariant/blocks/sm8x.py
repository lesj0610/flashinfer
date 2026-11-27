"""The SM8x bodies of the batch-invariant GEMM classes (SM80 / SM86 / SM87 / SM89).

The MMA step of each class is the architecture's own: the SM80 ``mma.sync`` k16 fold of the SM8x
dense GEMMs (`flashinfer.gemm.sm8x_fp8_dense`), one fp32 running sum per output element from +0 in
K order, whatever the row tile and M. The callables take the SM100 bodies' operand views and
arguments (`blocks.compile`), so the API's chunk structure stays the class's: the workspace split's
partials are raw fp32 sums (unit scales, ``x * 1`` is exact) and the in-CTA chunk fold is the same
partials folded by the split's fixup kernel in chunk order, the chain applied once after the fold.
The knobs and the persistent grid are placements of the SM100 bodies and are not read. The
block-scale class's body is `blocks.sm8x_fp8bs` (its own cache module).
"""

import torch


def gemm_callable(chunks: int, in_cta: bool, out_fmt: str):
    """The per-tensor fp8 class's body: ``run(a3, b3, c3, a_scale, b_scale)`` for the direct stores
    (the unsplit class, L = 1 or the batch; the in-CTA chunk fold, L = the chunks and ``c3`` of L 1)
    and ``run(a3, b3, ws)`` for the workspace split's partials (L = the chunks), with ``a3`` ``(M,
    Kc, L)``, ``b3`` ``(N, Kc, L)`` and ``c3`` ``(M, N, L)`` as `blocks.compile.get_compiled_gemm`
    hands them over."""
    from ...sm8x_fp4_dense import _ones
    from ...sm8x_fp8_dense import _run_fp8, row_tile
    from ..fixup import MAX_GRID_Y, get_fixup

    def run(a3, b3, c3, *scales):
        M, N = int(a3.shape[0]), int(b3.shape[0])
        if M == 0:
            return
        a, b, r = a3.permute(2, 0, 1), b3.permute(2, 1, 0), row_tile(M)
        if chunks == 1:
            _run_fp8(a, b, scales[0], scales[1], c3.permute(2, 0, 1), r, False)
            return
        one = _ones(a3.device)
        if not in_cta:  # the workspace split: the raw fp32 partials
            _run_fp8(a, b, one, one, c3.permute(2, 0, 1), r, False)
            return
        ws = torch.empty((chunks, M, N), dtype=torch.float32, device=a3.device)
        _run_fp8(a, b, one, one, ws, r, False)
        get_fixup(chunks, out_fmt, True, False)(
            ws.permute(1, 2, 0), c3[:, :, 0], *scales, M, N, min(M, MAX_GRID_Y)
        )

    return run
