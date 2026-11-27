"""The SM8x body of the block-scaled classes ``gemm.mxfp8`` / ``gemm.nvfp4`` / ``gemm.mxfp4`` (SM80 / SM86 / SM87 / SM89).

The MMA step is the architecture's own: the SM8x dense GEMMs' block-scaled fold (`flashinfer.gemm.sm8x_fp8_dense`'s
MXFP8 path, `flashinfer.gemm.sm8x_fp4_dense`'s NVFP4 / MXFP4 path), one fp32 running sum per output element from +0 in K
order with the block scales of each block applied, whatever the row tile and M. The callable takes the SM100 body's
operands and call forms (`blocks.compile_blockscaled`, ``api._run_mx``), so the chunk structure stays the class's: a
chunk's partial is the same fold over its K range from +0 (its scale-factor atom columns), the workspace split stores the
raw fp32 partials, and the in-CTA chunk fold and the in-cluster split fold the same partials with the split's fixup kernel
in chunk order, the chain (cvt; the 4-bit classes' ``alpha`` first) applied once after the fold. The knobs, the persistent
grid and the swapped decode record are placements of the SM100 body and are not read for the bits.
"""

from ..spec import A_ELEM_BITS_OF, SF_ATOM_KBLOCKS, SF_ATOM_ROWS, SF_VEC


def blockscaled_callable(chunks: int, swap_ab: bool, in_fmt: str, out_fmt: str):
    """``run(a3, b3, sfa, sfb, c3, *alpha)`` with the operands of ``api._run_mx``: A ``(M, Kb, L)`` and B ``(N, Kb, L)`` in
    bytes along K, the flat 128x4 swizzled scale buffers, C ``(M, N, 1)`` (the direct stores) or the workspace's ``(M, N,
    C)`` view (the split's partials); with ``swap_ab`` the unsplit class and the split's partials come exchanged (the
    weight as A, the output's transposed view as C)."""
    import torch

    # the fixup's objects are the dense module's and keyed there; this body compiles nothing in the block-scaled module
    from flashinfer.gemm.batch_invariant.fixup import MAX_GRID_Y, get_fixup

    from ...sm8x_fp4_dense import _run_fp4
    from ...sm8x_fp8_dense import _run_fp8, row_tile

    per_byte = 8 // A_ELEM_BITS_OF[in_fmt]  # elements per byte along K (both operands)
    atom_k = SF_ATOM_KBLOCKS * SF_VEC[in_fmt]  # K elements per scale-factor atom column
    atom_bytes = SF_ATOM_ROWS * SF_ATOM_KBLOCKS

    def gemm(a, b, sfa, sfb, out, alpha):
        """One block-scaled product: ``a`` ``(M, Kb)``, ``b`` ``(N, Kb)``, ``out`` ``(M, N)``."""
        r = row_tile(int(a.shape[0]))
        if in_fmt == "mxfp8":
            _run_fp8(a[None], b.t()[None], sfa, sfb, out[None], r, False, True)
        else:
            _run_fp4(a, b.t(), sfa, sfb, alpha, out, r, in_fmt == "mxfp4")

    def atoms(sf, rows, k, c, nkc):
        """Chunk ``c``'s scale-factor atom columns (``nkc`` of them) of a ``(rows, k)`` operand, as their own buffer."""
        rt, nka = -(-rows // SF_ATOM_ROWS), -(-k // atom_k)
        sf = sf.reshape(-1)[: rt * nka * atom_bytes].view(rt, nka, atom_bytes)
        return sf[:, c * nkc : (c + 1) * nkc].contiguous().view(-1)

    def run(a3, b3, sfa, sfb, c3, *alpha):
        # the workspace split's partials: C of L = the chunks (the direct stores' C has L = 1)
        split = chunks > 1 and int(c3.shape[2]) == chunks
        if swap_ab and (chunks == 1 or split):  # the exchanged operands back in place
            a3, b3, sfa, sfb, c3 = b3, a3, sfb, sfa, c3.permute(1, 0, 2)
        M, N = int(a3.shape[0]), int(b3.shape[0])
        if M == 0:
            return
        # the whole K in bytes (L = 1, or the chunks of contiguous K-major rows)
        Kb = int(a3.shape[1]) * int(a3.shape[2])
        a = a3.as_strided((M, Kb), (a3.stride(0), 1))
        b = b3.as_strided((N, Kb), (b3.stride(0), 1))
        if chunks == 1:
            gemm(a, b, sfa, sfb, c3[:, :, 0], alpha[0] if alpha else None)
            return
        K, Kcb = Kb * per_byte, Kb // chunks
        nkc = K // chunks // atom_k
        if split:
            ws = c3.permute(2, 0, 1)  # chunk-major (C, M, N)
        else:
            ws = torch.empty((chunks, M, N), dtype=torch.float32, device=a3.device)
        for c in range(chunks):  # the raw partials: block scales applied, no alpha
            ks = slice(c * Kcb, (c + 1) * Kcb)
            sa, sb = atoms(sfa, M, K, c, nkc), atoms(sfb, N, K, c, nkc)
            gemm(a[:, ks], b[:, ks], sa, sb, ws[c], None)
        if not split:  # the in-CTA chunk fold and the in-cluster split: the fixup's fold, then the chain
            fixup = get_fixup(chunks, out_fmt, alpha_given=bool(alpha))
            fixup(ws.permute(1, 2, 0), c3[:, :, 0], *alpha, M, N, min(M, MAX_GRID_Y))

    return run
