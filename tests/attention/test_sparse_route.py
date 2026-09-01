"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer

DEV = "cuda:0"

requires_cuda_sm80 = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 8,
    reason="the FA2 large-head path starts at sm_80",
)


def _expand_reference(
    block_indices, query_positions, sequence_lengths, token_to_req, compress_ratio
):
    """Expand the selection one row at a time, in plain Python."""
    rows, block_topk = block_indices.shape
    width = block_topk * compress_ratio + compress_ratio - 1
    num_requests = sequence_lengths.shape[0]
    out = torch.full((rows, width), -1, dtype=block_indices.dtype, device=DEV)
    blocks = block_indices.tolist()
    positions = query_positions.tolist()
    lengths = sequence_lengths.tolist()
    requests = token_to_req.tolist()
    for row in range(rows):
        request = requests[row]
        seq = lengths[request] if 0 <= request < num_requests else 0
        position = positions[row]
        complete = min(
            (position + 1) // compress_ratio, seq // compress_ratio, block_topk
        )
        route = []
        for rank in range(complete):
            base = blocks[row][rank] * compress_ratio
            route.extend(base + offset for offset in range(compress_ratio))
        tail_start = ((position + 1) // compress_ratio) * compress_ratio
        tail = min((position + 1) - tail_start, compress_ratio - 1)
        route.extend(tail_start + offset for offset in range(tail))
        for column, token in enumerate(route[:width]):
            if 0 <= token < seq:
                out[row, column] = token
    return out


def _route_reference(
    logical, token_to_req, block_table, page_size, num_slots, valid_rows
):
    """Map the logical route to slots and pack validity, one row at a time."""
    rows, width = logical.shape
    nbytes = -(-width // 8)
    route = torch.zeros((rows, width), dtype=logical.dtype, device=DEV)
    mask = torch.zeros(rows * nbytes, dtype=torch.uint8, device=DEV)
    table = block_table.tolist()
    requests = token_to_req.tolist()
    tokens = logical.tolist()
    table_width = block_table.shape[1]
    for row in range(rows):
        if row >= valid_rows:
            continue
        request = requests[row]
        if not 0 <= request < block_table.shape[0]:
            continue
        for column in range(width):
            token = tokens[row][column]
            if token < 0:
                continue
            page = token // page_size
            if page >= table_width:
                continue
            mapped = table[request][page]
            if mapped < 0:
                continue
            slot = mapped * page_size + token % page_size
            if slot >= num_slots:
                continue
            route[row, column] = slot
            mask[row * nbytes + column // 8] |= 1 << (column % 8)
    return route, mask


def _make_case(
    rows, block_topk, compress_ratio, seq_len, num_requests, page_size, seed
):
    g = torch.Generator(device=DEV).manual_seed(seed)
    blocks = torch.randint(
        0,
        max(1, seq_len // compress_ratio),
        (rows, block_topk),
        dtype=torch.int32,
        device=DEV,
        generator=g,
    )
    positions = torch.randint(
        0, seq_len, (rows,), dtype=torch.int32, device=DEV, generator=g
    )
    lengths = torch.full((num_requests,), seq_len, dtype=torch.int32, device=DEV)
    token_to_req = torch.randint(
        0, num_requests, (rows,), dtype=torch.int32, device=DEV, generator=g
    )
    pages_per_request = (seq_len + page_size - 1) // page_size
    pages = pages_per_request * num_requests
    table = (
        torch.randperm(pages, device=DEV, generator=g)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    # unmapped pages the validity has to catch
    table[:, ::7] = -1
    return blocks, positions, lengths, token_to_req, table, pages * page_size


@pytest.mark.parametrize("compress_ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 64, 300])
@pytest.mark.parametrize("block_topk", [1, 16, 128])
def test_expand_block_route(compress_ratio, rows, block_topk):
    blocks, positions, lengths, token_to_req, _, _ = _make_case(
        rows, block_topk, compress_ratio, 512, 4, 64, seed=rows + block_topk
    )
    out = flashinfer.expand_block_route(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    expected = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_expand_block_route_empties_a_row_without_a_request():
    blocks = torch.zeros(2, 4, dtype=torch.int32, device=DEV)
    positions = torch.tensor([100, 100], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int32, device=DEV)
    token_to_req = torch.tensor([0, -1], dtype=torch.int32, device=DEV)
    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 4)
    assert int((out[1] >= 0).sum()) == 0
    assert int((out[0] >= 0).sum()) > 0


@pytest.mark.parametrize("compress_ratio", [2, 4])
@pytest.mark.parametrize("rows", [1, 9, 128])
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_qsa_route_from_blocks(compress_ratio, rows, page_size):
    block_topk = 32
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 1024, 4, page_size, seed=rows + page_size
    )
    width = block_topk * compress_ratio + compress_ratio - 1
    nbytes = -(-width // 8)
    logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)

    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        compress_ratio,
        page_size,
        num_slots,
    )

    expected_logical = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    torch.testing.assert_close(logical, expected_logical, rtol=0, atol=0)
    expected_route, expected_mask = _route_reference(
        expected_logical, token_to_req, table, page_size, num_slots, rows
    )
    torch.testing.assert_close(route, expected_route, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("rows,valid_rows", [(1, 1), (16, 16), (128, 100), (64, 0)])
@pytest.mark.parametrize("page_size", [1, 64])
def test_qsa_route_from_logical(rows, valid_rows, page_size):
    """Padding rows must come out fully masked, whatever the logical route holds."""
    block_topk, compress_ratio = 32, 4
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 1024, 4, page_size, seed=rows
    )
    logical = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    width = logical.shape[1]
    nbytes = -(-width // 8)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)

    flashinfer.qsa_route_from_logical(
        logical, token_to_req, table, route, mask, valid_rows, page_size, num_slots
    )
    expected_route, expected_mask = _route_reference(
        logical, token_to_req, table, page_size, num_slots, valid_rows
    )
    torch.testing.assert_close(route, expected_route, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)


def test_qsa_route_never_points_outside_the_cache():
    """An entry the mask clears must still hold an in-range slot."""
    rows, block_topk, compress_ratio, page_size = 32, 16, 4, 16
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 256, 2, page_size, seed=7
    )
    # every page unmapped: nothing is valid, and every slot must still be legal
    table.fill_(-1)
    width = block_topk * compress_ratio + compress_ratio - 1
    nbytes = -(-width // 8)
    logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)
    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        compress_ratio,
        page_size,
        num_slots,
    )
    assert int(mask.sum()) == 0
    assert int(route.min()) >= 0
    assert int(route.max()) < num_slots


def test_route_ops_reject_bad_arguments():
    blocks = torch.zeros(4, 8, dtype=torch.int32, device=DEV)
    positions = torch.zeros(4, dtype=torch.int32, device=DEV)
    lengths = torch.full((1,), 64, dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(4, dtype=torch.int32, device=DEV)

    with pytest.raises(ValueError, match="compress_ratio"):
        flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 0)
    with pytest.raises(ValueError, match="2D"):
        flashinfer.expand_block_route(blocks[0], positions, lengths, token_to_req, 4)
    with pytest.raises(ValueError, match="shape"):
        flashinfer.expand_block_route(
            blocks,
            positions,
            lengths,
            token_to_req,
            4,
            out=torch.zeros(4, 8, dtype=torch.int32, device=DEV),
        )
    # a ratio the dispatch has no kernel for
    with pytest.raises(Exception, match="compress_ratio"):
        flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 3)


@requires_cuda_sm80
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_route_reads_a_wide_quantized_head(head_dim, kv_dtype):
    """A one-byte cache of any supported width reads on this architecture.

    The FA2 large-head path rebuilds a quantized cache from raw bytes before
    the dots, which does not depend on the architecture the bytes were written
    for. This pins that down for the widths the read paths claim.
    """
    num_qo_heads, num_kv_heads, page_size = 8, 1, 64
    rows, width, pages = 4, 64, 8
    g = torch.Generator(device=DEV).manual_seed(head_dim)

    keys = (
        torch.randn(
            pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=DEV,
            generator=g,
        )
        * 0.3
    )
    values = torch.randn_like(keys) * 0.3
    scale = 0.5
    if kv_dtype == torch.float8_e4m3fn:
        keys = (keys.float() / scale).to(kv_dtype)
        values = (values.float() / scale).to(kv_dtype)
        run_kwargs = {"k_scale": scale, "v_scale": scale}
        # The reference reads the same bytes, dequantized ahead of time.
        ref_keys = (keys.float() * scale).to(torch.bfloat16)
        ref_values = (values.float() * scale).to(torch.bfloat16)
    else:
        run_kwargs = {}
        ref_keys, ref_values = keys, values

    query = torch.randn(
        rows, num_qo_heads, head_dim, dtype=torch.bfloat16, device=DEV, generator=g
    )
    route = torch.randint(
        0, pages * page_size, (rows, width), dtype=torch.int32, device=DEV, generator=g
    )
    indptr = torch.arange(0, (rows + 1) * width, width, dtype=torch.int32, device=DEV)
    mask = torch.ones(rows * width, 1, 1, dtype=torch.bool, device=DEV)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    def run(k, v, dtype, **kwargs):
        wrapper = flashinfer.BlockSparseAttentionWrapper(workspace, kv_layout="HND")
        wrapper.plan(
            indptr,
            route.reshape(-1).contiguous(),
            rows,
            pages * page_size,
            1,
            1,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            q_data_type=torch.bfloat16,
            kv_data_type=dtype,
            o_data_type=torch.bfloat16,
            kv_cache_page_size=page_size,
        )
        return wrapper.run(query, k, v, **kwargs)

    out = run(keys, values, kv_dtype, **run_kwargs)
    expected = run(ref_keys, ref_values, torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def _scores_case(head_dim=128, num_heads=4, rows=8, pages=6, page_size=8, dtype=torch.bfloat16):
    """An aligned scorer case, plus what it takes to re-run it from a view."""
    torch.manual_seed(0)
    q = torch.randn(rows, num_heads, head_dim, dtype=dtype, device=DEV)
    k_cache = torch.randn(pages, page_size, head_dim, dtype=dtype, device=DEV)
    page_table = torch.arange(pages, dtype=torch.int32, device=DEV).reshape(1, pages)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=DEV)
    positions = torch.arange(rows, dtype=torch.int32, device=DEV) + pages * page_size
    seq_lens = torch.full((1,), pages * page_size, dtype=torch.int32, device=DEV)
    return q, k_cache, page_table, token_to_req, positions, seq_lens


def _run_scores(q, k_cache, page_table, token_to_req, positions, seq_lens):
    return flashinfer.sparse_paged_scores(
        q, k_cache, page_table, token_to_req, positions, seq_lens, 1, q.shape[2] ** 0.5
    )


@requires_cuda_sm80
def test_scores_accept_a_query_view_with_a_storage_offset():
    """A view onto the middle of a tensor keeps its strides and loses its
    alignment, which the staged 128-bit loads cannot assume away."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    head_dim = q.shape[2]
    backing = torch.empty(
        q.shape[0], q.shape[1], head_dim + 1, dtype=q.dtype, device=DEV
    )
    backing[..., 1:] = q
    offset_q = backing[..., 1:]
    assert offset_q.storage_offset() != 0
    got, got_visible = _run_scores(offset_q, k_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)


@requires_cuda_sm80
def test_scores_accept_a_cache_view_with_a_storage_offset():
    """The same for the cache, which is staged through cp_async."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    pages, page_size, head_dim = k_cache.shape
    backing = torch.empty(
        pages, page_size, head_dim + 1, dtype=k_cache.dtype, device=DEV
    )
    backing[..., 1:] = k_cache
    offset_cache = backing[..., 1:]
    assert offset_cache.storage_offset() != 0
    got, got_visible = _run_scores(q, offset_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)
