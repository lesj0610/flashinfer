"""
Copyright (c) 2025 by FlashInfer team.

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

# ruff: noqa: B008

import os
import random
import math

import pytest
import torch

from .reference_delta_rule import exclusive_cumsum
from . import reference_delta_rule as reference
from flashinfer.utils import (
    is_sm8x_supported,
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)

if torch.cuda.is_available() and is_sm90a_supported(torch.device("cuda")):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
        cp_delta_rule_dsl_sm90 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm90 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm90 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm90 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm90 as cp_delta_rule_t_precompute_dsl,
    )
elif (
    torch.cuda.is_available()
    and is_sm100a_supported(torch.device("cuda"))
    and torch.version.cuda is not None
    and int(torch.version.cuda.split(".")[0]) >= 13
):
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm100 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm100 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm100 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm100 as cp_delta_rule_t_precompute_dsl,
    )
elif torch.cuda.is_available() and is_sm12x_supported(torch.device("cuda")):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120 import (
        cp_delta_rule_dsl_sm120 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm120 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm120 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm120 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm120 as cp_delta_rule_t_precompute_dsl,
    )
elif torch.cuda.is_available() and is_sm8x_supported(torch.device("cuda")):
    # Ported one stage at a time. What is not here yet stays None, so the tests
    # for it skip rather than reaching for a name that does not exist -- and a
    # half-ported pipeline is never assembled by accident.
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_dsl_sm80 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm80 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm80 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm80 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm80 as cp_delta_rule_t_precompute_dsl,
    )
else:
    cp_delta_rule_dsl = None
    cp_delta_rule_fixup_dsl = None
    cp_delta_rule_mn_precompute_dsl = None
    cp_delta_rule_prefill_dsl = None
    cp_delta_rule_t_precompute_dsl = None

from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
    chunk_bound_host,
    workspace_num_chunks_host,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule


FIXUP_TF32_ATOL = 2e-3
FIXUP_TF32_RTOL = 2e-3
FIXUP_KERNEL_KINDS = ["simt_row4", "simt_row8", "hmma"]
if torch.cuda.is_available() and is_sm8x_supported(torch.device("cuda")):
    # The HMMA fixup hands its math warps extra registers with `setmaxnreg`,
    # which SM8x does not have. It is not built there, so there is no kind to
    # parametrize over rather than a kind that raises.
    FIXUP_KERNEL_KINDS = ["simt_row4", "simt_row8"]


def _skip_if_cp_unsupported(*stages):
    """Skip when this device has no CP path, or none for the stages asked for.

    SM8x is being ported a stage at a time, so a test names the entry points it
    needs and skips when one of them is still None. Naming them is what keeps a
    half-ported pipeline from being assembled: the test for a stage that does
    not exist skips, it does not fail on a missing attribute and it does not
    quietly run against another architecture's.
    """
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 CP GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
    elif is_sm8x_supported(device):
        pass
    elif not (is_sm90a_supported(device) or is_sm12x_supported(device)):
        pytest.skip("CP GDN prefill requires SM8x, SM90, SM100, or SM12x")

    for stage in stages or ("cp_delta_rule_dsl",):
        if globals().get(stage) is None:
            pytest.skip(f"{stage} is not ported for this device yet")


def _seed_all(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _make_cu_seqlens(seq_lens, device):
    return torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)


def _make_gates(total_seqlen, num_heads, baseline, device):
    return (
        baseline
        + (1.0 - baseline)
        * torch.rand(total_seqlen, num_heads, dtype=torch.float32, device=device)
    ).contiguous()


@torch.inference_mode()
def _run_cp_kernel_chain(
    q,
    k,
    v,
    alpha,
    beta,
    cu_seqlens,
    total_seqlen,
    max_seqlen,
    cp_chunk_len,
    scale,
    initial_state=None,
):
    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
    )

    our_o = torch.empty(
        (total_seqlen, max(q.shape[1], v.shape[1]), q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    our_state = torch.empty(
        cu_seqlens.numel() - 1,
        max(q.shape[1], v.shape[1]),
        q.shape[2],
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        initial_state=initial_state,
    )
    return our_o, our_state


@torch.inference_mode()
def _run_non_cp_prefill(q, k, v, alpha, beta, cu_seqlens, scale, initial_state=None):
    ref_o = torch.empty(
        (q.shape[0], max(q.shape[1], v.shape[1]), q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    ref_state = torch.empty(
        cu_seqlens.numel() - 1,
        max(q.shape[1], v.shape[1]),
        q.shape[2],
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        initial_state,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    return ref_o, ref_state


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[128], [192, 64], [2048], [1025], [9999, 6553]])
@pytest.mark.parametrize("gate_baseline", [1.0, 0.9, 0.9995])
def test_cp_delta_rule_t_precompute(
    qkv_factory,
    dtype,
    seq_lens,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_t_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    num_heads = 1
    head_size = 128
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, _ = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    our_t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    torch.cuda.synchronize()

    assert our_t.shape == (
        workspace_num_chunks_host(cu_seqlens.cpu(), 64, total_seqlen),
        num_heads,
        64,
        64,
    )
    for seq_idx, _ in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        t_start = chunk_bound_host(seq_idx, seq_start, 64)
        ref_t = reference.precompute_blockwise_cp_delta_rule_t(
            k[seq_start:seq_end],
            beta[seq_start:seq_end],
            block_size=64,
            kv_dtype=torch.float32,
            t_dtype=dtype,
        )
        torch.testing.assert_close(
            our_t[t_start : t_start + ref_t.shape[0]], ref_t, atol=5e-3, rtol=5e-3
        )


@torch.inference_mode()
def test_cp_delta_rule_t_precompute_varlen_tail_is_projected(
    qkv_factory,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_t_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 1
    head_size = 128
    seq_lens = [96]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, _ = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)

    got = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    torch.cuda.synchronize()

    tail = got[1, 0]
    torch.testing.assert_close(
        tail[32:], torch.zeros_like(tail[32:]), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        tail[:, 32:], torch.zeros_like(tail[:, 32:]), atol=0.0, rtol=0.0
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("gate_baseline", [0.9, 0.99, 0.9995])
@pytest.mark.parametrize(
    "seq_lens, cp_chunk_len",
    [
        ([64, 192], 64),
        ([128, 200], 128),
        ([192], 192),
        ([1024, 3000], 1024),
        ([96, 64, 192], 128),
    ],
)
def test_cp_delta_rule_mn_precompute(
    qkv_factory,
    dtype,
    seq_lens,
    cp_chunk_len,
    num_heads,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_mn_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    head_size = 128
    block_size = 64
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    our_transfer, our_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    assert our_transfer.shape == (
        workspace_num_chunks_host(cu_seqlens.cpu(), cp_chunk_len, total_seqlen),
        num_heads,
        head_size,
        head_size,
    )
    assert our_state.shape == our_transfer.shape

    for seq_idx, seq_len in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        cp_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        t_start = chunk_bound_host(seq_idx, seq_start, block_size)
        num_cp_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        for chunk_idx in range(num_cp_chunks):
            chunk_offset = chunk_idx * cp_chunk_len
            chunk_end = min(seq_len, chunk_offset + cp_chunk_len)
            num_t_blocks = (chunk_end - chunk_offset + block_size - 1) // block_size
            t_block_offset = chunk_idx * (cp_chunk_len // block_size)
            slot = cp_start + chunk_idx
            ref_transfer, ref_state = reference.blockwise_cp_delta_rule_pre_transposed(
                k[seq_start + chunk_offset : seq_start + chunk_end],
                v[seq_start + chunk_offset : seq_start + chunk_end],
                alpha[seq_start + chunk_offset : seq_start + chunk_end],
                t[t_start + t_block_offset : t_start + t_block_offset + num_t_blocks],
                block_size=block_size,
                kv_dtype=torch.float32,
            )
            if dtype == torch.bfloat16:
                atol = 5e-3
                rtol = 2e-3
            else:
                atol = 1e-3
                rtol = 5e-4
            torch.testing.assert_close(
                our_transfer[slot].transpose(-1, -2), ref_transfer, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                our_state[slot].transpose(-1, -2), ref_state, atol=atol, rtol=rtol
            )


@torch.inference_mode()
@pytest.mark.parametrize("kernel_kind", FIXUP_KERNEL_KINDS)
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("num_heads", [1, 3])
@pytest.mark.parametrize(
    "seq_lens, cp_chunk_len", [([96, 0, 300], 128), ([1], 1), ([2], 1), ([5], 1)]
)
def test_cp_delta_rule_fixup(
    seq_lens,
    cp_chunk_len,
    num_heads,
    use_initial_state,
    kernel_kind,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_fixup_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    total_seqlen = sum(seq_lens)
    total_cp_chunks = workspace_num_chunks_host(
        cu_seqlens.cpu(), cp_chunk_len, total_seqlen
    )

    local_transfer = (
        torch.randn(
            total_cp_chunks,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    local_state = (
        torch.randn(
            total_cp_chunks,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    diag = torch.arange(head_size, device=device)
    local_transfer[:, :, diag, diag] += 0.9
    initial_state = None
    if use_initial_state:
        initial_state = (
            torch.randn(
                len(seq_lens),
                num_heads,
                head_size,
                head_size,
                dtype=torch.float32,
                device=device,
            )
            * 0.03
        )

    our_fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer.contiguous(),
        local_state.contiguous(),
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
        _kernel_kind=kernel_kind,
    )
    torch.cuda.synchronize()

    ref_transfers_by_seq = []
    ref_states_by_seq = []
    ref_initial_states_by_seq = []
    ref_seq_indices = []
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        chunk_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        num_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        seq_slots = []
        for chunk_idx in range(num_chunks):
            slot = chunk_start + chunk_idx
            seq_slots.append(slot)
        if seq_slots:
            ref_transfers_by_seq.append(local_transfer[seq_slots])
            ref_states_by_seq.append(local_state[seq_slots])
            if use_initial_state:
                ref_initial_states_by_seq.append(initial_state[seq_idx])
            ref_seq_indices.append(seq_idx)

    _, ref_by_seq = reference.cp_delta_rule_fixup_transposed(
        ref_transfers_by_seq,
        ref_states_by_seq,
        ref_initial_states_by_seq if use_initial_state else None,
    )
    for seq_idx, seq_fixed in zip(ref_seq_indices, ref_by_seq):  # noqa: B905
        seq_start = int(cu_seqlens[seq_idx].item())
        chunk_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        for chunk_idx in range(seq_fixed.shape[0]):
            torch.testing.assert_close(
                our_fixed_state[chunk_start + chunk_idx],
                seq_fixed[chunk_idx],
                atol=FIXUP_TF32_ATOL,
                rtol=FIXUP_TF32_RTOL,
            )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("chunk_len", [64, 128])
@pytest.mark.parametrize("gate_baseline", [0.9, 0.9995])
def test_cp_delta_rule_prefill_varlen_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    chunk_len,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    num_heads = 1
    head_size = 128
    cp_chunk_len = chunk_len
    seq_lens = [chunk_len, chunk_len]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_values = [0]
    for seq_len in seq_lens:
        cu_values.append(cu_values[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int64, device=device)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    q = q.contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer, local_state, cu_seqlens, total_seqlen, cp_chunk_len=cp_chunk_len
    )

    our_o = torch.empty_like(q)
    our_state = torch.empty(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(q)
    ref_state = torch.empty(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads", [(4, 1, 1), (1, 1, 4)]
)
def test_cp_delta_rule_prefill_varlen_matches_non_cp_prefill_unequal_heads(
    qkv_factory,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_size = 128
    cp_chunk_len = 64
    seq_lens = [96, 64]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_values = [0]
    for seq_len in seq_lens:
        cu_values.append(cu_values[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int64, device=device)
    num_sab_heads = max(num_q_heads, num_v_heads)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    q = q.contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_sab_heads, 0.99, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer, local_state, cu_seqlens, total_seqlen, cp_chunk_len=cp_chunk_len
    )

    our_o = torch.empty(
        total_seqlen, num_sab_heads, head_size, dtype=dtype, device=device
    )
    our_state = torch.empty(
        len(seq_lens),
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(our_o)
    ref_state = torch.empty_like(our_state)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize(
    "dtype, seq_lens, cp_chunk_len, num_q_heads, num_k_heads, num_v_heads, gate_baseline, scale",
    [
        (torch.bfloat16, [2048], 1024, 1, 1, 1, 0.99, 1.0),
        (torch.bfloat16, [4096], 2048, 2, 1, 1, 0.9995, 1.0),
        (torch.float16, [2049], 1024, 1, 1, 1, 0.99, "auto"),
        (torch.float16, [8193], 2048, 1, 1, 1, 0.99, "auto"),
        (torch.bfloat16, [1536, 257], 1024, 1, 1, 2, 0.99, 1.0),
    ],
)
def test_cp_delta_rule_kernel_chain_long_small_bh_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    seq_lens,
    cp_chunk_len,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    gate_baseline,
    scale,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    num_sab_heads = max(num_q_heads, num_v_heads)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)

    our_o, our_state = _run_cp_kernel_chain(
        q, k, v, alpha, beta, cu_seqlens, total_seqlen, max_seqlen, cp_chunk_len, scale
    )
    torch.cuda.synchronize()
    ref_o, ref_state = _run_non_cp_prefill(q, k, v, alpha, beta, cu_seqlens, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(our_state, ref_state, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[192, 64], [1025]])
def test_cp_delta_rule_e2e_with_initial_state(
    qkv_factory,
    dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0
    dtype = getattr(torch, dtype)
    num_heads = 1

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)
    initial_state = (
        torch.randn(
            len(seq_lens),
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )

    our_o = torch.empty(
        [total_seqlen, num_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty_like(initial_state)
    cp_delta_rule_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        scale,
        initial_state=initial_state,
        max_seqlen=max(seq_lens),
        cp_chunk_len=128,
    )
    torch.cuda.synchronize()

    ref_o, ref_state = _run_non_cp_prefill(
        q, k, v, alpha, beta, cu_seqlens, scale, initial_state=initial_state
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(our_state, ref_state, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("gate_baseline", [0.9995, 0.99])
@pytest.mark.parametrize("num_heads", [(1, 1, 1), (2, 2, 8), (4, 1, 1), (16, 16, 64)])
@pytest.mark.parametrize("seq_lens", [[192, 64], [2048], [1025], [9999, 65530]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_cp_delta_rule_e2e(
    qkv_factory,
    dtype,
    seq_lens,
    num_heads,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0
    dtype = getattr(torch, dtype)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_heads)

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)

    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty(
        len(seq_lens),
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    our_o.fill_(float("nan"))
    our_state.fill_(float("nan"))
    cp_delta_rule_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        scale,
        max_seqlen=max(seq_lens),
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(our_o)
    ref_state = torch.empty_like(our_state)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    if dtype == torch.bfloat16:
        ref_o = ref_o.to(dtype)
        atol_o = 2e-2
        rtol_o = 2e-2
        atol_state = 1e-2
        rtol_state = 5e-3
    else:
        atol_o = 5e-3
        rtol_o = 5e-3
        atol_state = 1e-3
        rtol_state = 1e-3

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_state, rtol=rtol_state)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[128], [256, 64], [2048]])
def test_cp_delta_rule_public_wrapper_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    head_size = 128
    num_heads = 1
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)

    our_o, our_state = chunk_gated_delta_rule(
        q, k, v, alpha, beta, scale, None, True, cu_seqlens, True, use_cp=True
    )
    ref_o, ref_state = chunk_gated_delta_rule(
        q, k, v, alpha, beta, scale, None, True, cu_seqlens, True, use_cp=False
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[128], [256, 64]])
def test_cp_delta_rule_external_state_dtype(
    qkv_factory,
    state_dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    device = torch.device("cuda")
    _seed_all(seed)
    dtype = torch.bfloat16
    head_size = 128
    num_heads = 1
    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with device:
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
        initial_state = (
            torch.randn(num_seqs, num_heads, head_size, head_size) * 0.01
        ).to(state_dtype)
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)
    our_o = torch.empty_like(q)
    ref_o = torch.empty_like(q)
    our_state = torch.empty_like(initial_state)
    ref_state = torch.empty_like(initial_state)

    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,
        initial_state,
        True,
        cu_seqlens,
        False,
        output=our_o,
        output_state=our_state,
        use_cp=True,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,
        initial_state,
        True,
        cu_seqlens,
        False,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


# ─── Regressions for the CP prefill's bookkeeping ─────────────────────────────
# Three properties, one per defect. None of them is a value comparison against a
# reference: they hold whether or not the recurrence itself is right, so they
# keep working while it is being fixed and they fail the moment the bookkeeping
# regresses.


def _cp_stage_inputs(
    seq_lens, cp_chunk_len, qkv_factory, device, dtype,
    num_q_heads=1, num_k_heads=1, num_v_heads=1,
):
    """Stages 1 through 3, and everything stage 4 needs to run.

    Head counts are separate parameters because they arrive at the kernel as
    four adjacent scalars, and an equal-head case cannot tell them apart -- the
    argument shift this file's tests were written for was invisible in exactly
    that way.
    """
    head_size = 128
    num_heads = max(num_q_heads, num_v_heads)
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    beta = _make_gates(total_seqlen, num_heads, 0.25, device)
    alpha = _make_gates(total_seqlen, num_heads, 0.9, device)
    k_sab = k if num_k_heads == num_heads else k.repeat_interleave(
        num_heads // num_k_heads, dim=1
    ).contiguous()
    max_seqlen = max(seq_lens)
    t = cp_delta_rule_t_precompute_dsl(
        k_sab, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k_sab, v, t, alpha, cu_seqlens, total_seqlen,
        cp_chunk_len=cp_chunk_len, max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer, local_state, cu_seqlens, total_seqlen,
        cp_chunk_len=cp_chunk_len,
    )
    return dict(
        q=q, k=k, v=v, t=t, alpha=alpha, fixed_state=fixed_state,
        cu_seqlens=cu_seqlens, total_seqlen=total_seqlen, max_seqlen=max_seqlen,
        num_heads=num_heads, head_size=head_size, cp_chunk_len=cp_chunk_len,
        beta=beta,
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cp_prefill_leaves_the_fixup_workspace_alone(qkv_factory, dtype):
    """No final state asked for means none written.

    `fixed_state` is the fixup workspace, and on this path the state tensor the
    kernel would write points into it. A store that runs when it was not asked
    for lands on a chunk state another block still has to read, so the property
    is that the workspace comes back byte for byte.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    seq_lens = [64, 192]
    args = _cp_stage_inputs(seq_lens, 64, qkv_factory, device, getattr(torch, dtype))
    before = args["fixed_state"].clone()
    o = torch.empty(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        dtype=args["q"].dtype, device=device,
    )
    cp_delta_rule_prefill_dsl(
        o, None, args["q"], args["k"], args["v"], args["t"], args["fixed_state"],
        args["alpha"], 1.0, args["cu_seqlens"], args["total_seqlen"],
        cp_chunk_len=args["cp_chunk_len"], max_seqlen=args["max_seqlen"],
    )
    torch.cuda.synchronize()
    assert torch.equal(args["fixed_state"], before), (
        "the fixup workspace was written while no final state was asked for"
    )


@torch.inference_mode()
# The kernel takes GQA (k == v, q a multiple) and GVA (q == k, v a multiple),
# so those are the two ways the four head counts can differ from each other.
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(1, 1, 1), (4, 1, 1), (1, 1, 4)],
)
def test_cp_prefill_writes_every_value_band(
    qkv_factory, num_q_heads, num_k_heads, num_v_heads
):
    """Both halves of O get written.

    `head_size` is 128 and the fused kernel owns 64 rows per block, so two value
    slices cover it. A grid without that axis runs slice 0 only and leaves the
    other band at whatever the caller passed -- which a value comparison against
    a reference would report as "wrong numbers" rather than "never written".
    Poisoning O separates the two.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    args = _cp_stage_inputs(
        [64], 64, qkv_factory, device, torch.float16,
        num_q_heads=num_q_heads, num_k_heads=num_k_heads, num_v_heads=num_v_heads,
    )
    poison = float("nan")
    o = torch.full(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        poison, dtype=args["q"].dtype, device=device,
    )
    state = torch.zeros(
        len(args["cu_seqlens"]) - 1, args["num_heads"],
        args["head_size"], args["head_size"], dtype=torch.float32, device=device,
    )
    cp_delta_rule_prefill_dsl(
        o, state, args["q"], args["k"], args["v"], args["t"], args["fixed_state"],
        args["alpha"], 1.0, args["cu_seqlens"], args["total_seqlen"],
        cp_chunk_len=args["cp_chunk_len"], max_seqlen=args["max_seqlen"],
    )
    torch.cuda.synchronize()
    half = args["head_size"] // 2
    for band, name in ((o[..., :half], "band 0"), (o[..., half:], "band 1")):
        assert not band.isnan().any(), f"{name} of O was never written"


@torch.inference_mode()
def test_cp_prefill_checkpoints_land_in_sequence_order(qkv_factory):
    """Every checkpoint slot holds the state after its own prefix.

    The shape matters. A `cp_chunk_len` equal to the block length runs one block
    per chunk and never reaches the middle-block or last-block checkpoint calls,
    so this is 384 tokens in chunks of 192 with a checkpoint every 64: two
    chunks, three blocks each, and the second chunk goes through both paths.

    Checking that a slot was written catches an index that collapses; checking
    it against the state after that many tokens also catches two that swapped.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    cp_chunk_len = 192
    every = 64
    seq_lens = [384]
    args = _cp_stage_inputs(seq_lens, cp_chunk_len, qkv_factory, device, torch.float16)
    per_seq = [length // every for length in seq_lens]
    starts = [0]
    for count in per_seq:
        starts.append(starts[-1] + count)
    checkpoints = torch.full(
        (sum(per_seq), args["num_heads"], args["head_size"], args["head_size"]),
        float("nan"), dtype=torch.float32, device=device,
    )
    o = torch.empty(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        dtype=args["q"].dtype, device=device,
    )
    state = torch.zeros(
        len(seq_lens), args["num_heads"], args["head_size"], args["head_size"],
        dtype=torch.float32, device=device,
    )
    cp_delta_rule_prefill_dsl(
        o, state, args["q"], args["k"], args["v"], args["t"], args["fixed_state"],
        args["alpha"], 1.0, args["cu_seqlens"], args["total_seqlen"],
        cp_chunk_len=cp_chunk_len, max_seqlen=args["max_seqlen"],
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=torch.tensor(starts, dtype=torch.int32, device=device),
        checkpoint_every_n_tokens=every,
    )
    torch.cuda.synchronize()
    for slot in range(checkpoints.shape[0]):
        assert not checkpoints[slot].isnan().any(), (
            f"checkpoint slot {slot} of {checkpoints.shape[0]} was never written"
        )
    for slot in range(checkpoints.shape[0]):
        n = (slot + 1) * every
        _, prefix_state = _run_non_cp_prefill(
            args["q"][:n], args["k"][:n], args["v"][:n],
            args["alpha"][:n], args["beta"][:n],
            _make_cu_seqlens([n], device), 1.0,
        )
        torch.testing.assert_close(
            checkpoints[slot], prefix_state[0], atol=5e-3, rtol=2e-3,
            msg=lambda m, slot=slot, n=n: (
                f"checkpoint slot {slot} is not the state after {n} tokens\n{m}"
            ),
        )
