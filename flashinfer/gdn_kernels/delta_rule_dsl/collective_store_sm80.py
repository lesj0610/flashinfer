"""SMEM to GMEM store for the sm_80 delta-rule prefill kernel.

The sm_90 path stores O with a bulk tensor copy issued by a dedicated store
warp: one thread starts the tile and the hardware walks the descriptor,
clamping at the tensor bound. Neither half survives here.

The copy does not, because ``CopyBulkTensorTileS2GOp`` and the non-tensor
``CopyBulkS2GOp`` are both sm_90+, so the store is an ordinary vectorized one
and the bound the descriptor carried becomes a predicate -- a row past this
sequence's end is simply not written. That also drops the spare tensor map the
sm_90 path keeps in global memory and rewrites whenever a packed sequence ends
mid-tile, along with the fence around mutating it.

The dedicated warp does not survive either, for a reason that has nothing to do
with stores: through the pipeline this kernel can build there is no way for one
warp to wait on another warp's cp.async -- see pipeline_sm80 for which mbarrier
operations the DSL refuses on this target -- so every thread gets the same work
and the store warp has no one left to receive tiles from. The store is block-wide now, which
is also why it needs no pipeline -- the barrier before it is what makes the
math warps' writes visible, and the barrier after is what frees the buffer.
"""

import cutlass
import cutlass.cute as cute


class CollectiveStoreSm80:
    """Writes O tiles from shared memory with predicated vector stores."""

    def __init__(self, blk_q: int, d: int, num_threads: int):
        self.BLK_Q = blk_q
        self.D = d
        self.num_threads = num_threads
        # Elements a thread moves per store. O is bf16 and d is contiguous, so
        # eight of them is one 16 B access -- the widest a thread can do.
        self.vec = 8
        if d % self.vec != 0:
            # A d this shape does not divide is not one this kernel is built
            # for; the caller's can_implement rejects it earlier. Falling back
            # to one element keeps the store correct if it ever gets here.
            self.vec = 1
        self.lanes_per_row = d // self.vec
        if num_threads % self.lanes_per_row != 0:
            raise ValueError(
                f"store needs a thread count divisible by {self.lanes_per_row} "
                f"(d={d} / vec={self.vec}), got {num_threads}"
            )
        self.rows_at_a_time = num_threads // self.lanes_per_row

    @cute.jit
    def run(
        self,
        sO: cute.Tensor,
        gO: cute.Tensor,
        work_desc,
        blk: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """Write one O tile, skipping rows past this sequence's last token.

        Both barriers belong to the caller's loop as much as to this store: the
        first is what makes the math warps' shared writes visible to the loads
        below, and the second is what lets the next block overwrite the buffer.
        Every thread in the block reaches both.
        """
        cute.arch.barrier()

        gO_head = gO[None, None, work_desc.o_head_idx(num_q_heads, num_v_heads)]
        tok_base = work_desc.tok_offset + blk * cutlass.Int32(self.BLK_Q)
        # One past the last token this sequence owns. The sm_90 path folds this
        # into the descriptor; here it gates the row.
        tok_end = work_desc.tok_offset + work_desc.seq_len

        lanes_per_row = cutlass.Int32(self.lanes_per_row)
        thread_row = tid // lanes_per_row
        thread_col = (tid % lanes_per_row) * cutlass.Int32(self.vec)

        for row_base in cutlass.range_constexpr(0, self.BLK_Q, self.rows_at_a_time):
            row = cutlass.Int32(row_base) + thread_row
            tok = tok_base + row
            if row < cutlass.Int32(self.BLK_Q) and tok < tok_end:
                for i in cutlass.range_constexpr(self.vec):
                    gO_head[thread_col + i, tok] = sO[thread_col + i, row]

        cute.arch.barrier()


__all__ = ["CollectiveStoreSm80"]
