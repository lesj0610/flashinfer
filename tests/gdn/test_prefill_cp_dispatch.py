"""What `use_cp="auto"` picks, per architecture, with no kernel running.

These pin dispatch, not numerics. Every kernel entry is replaced with a
recorder, so the calls here are shape and capability checks only -- which is
the point: the choice has to be checkable without a device of each generation.

They exist because a prepared-execution experiment once added a shape rule to
`chunk_gated_delta_rule` and wired it into the SM90/100/120 auto condition.
The rule needs a host-side maximum that no existing caller passes, so it read
None, returned False, and turned off Hopper/Blackwell auto CP for every caller
-- silently, since falling back to the fused kernel is not an error. Nothing
here would have let that through.
"""

import pytest
import torch

import flashinfer.gdn_prefill as gp


HEAD_SIZE = 128
CP_ENTRIES = ("cp_delta_rule_dsl_sm80", "cp_delta_rule_dsl_sm90",
              "cp_delta_rule_dsl_sm100", "cp_delta_rule_dsl_sm120")
FUSED_ENTRIES = ("chunk_gated_delta_rule_sm80", "chunk_gated_delta_rule_sm90",
                 "chunk_gated_delta_rule_sm100", "chunk_gated_delta_rule_sm120")


def _inputs(seq_lens, heads, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    total = sum(seq_lens)
    q = torch.randn(total, heads, HEAD_SIZE, dtype=dtype, device=dev)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.zeros(total, heads, dtype=torch.float32, device=dev)
    beta = torch.ones(total, heads, dtype=torch.float32, device=dev)
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()],
                      dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu)


def _dispatch(monkeypatch, *, arch, sm_count=80, **call):
    """Which entry ran, under a mocked capability. Returns its name."""
    fired = []

    def recorder(name):
        def f(*a, **kw):
            fired.append(name)
        return f

    monkeypatch.setattr(gp, "get_compute_capability", lambda dev: (arch, 0))
    monkeypatch.setattr(gp, "get_device_name", lambda dev: f"mock-sm{arch}0")
    monkeypatch.setattr(gp, "get_device_sm_count", lambda dev: sm_count)
    # The SM100 path refuses CUDA < 13 before looking at anything else, and
    # this runs wherever the test host's toolkit happens to be.
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    for name in CP_ENTRIES + FUSED_ENTRIES:
        monkeypatch.setattr(gp, name, recorder(name))
    # A recorder writes nothing into `output`, which is fine: nothing here
    # reads the result.
    gp.chunk_gated_delta_rule(**call)
    assert len(fired) == 1, f"expected one entry, got {fired}"
    return fired[0]


@pytest.mark.parametrize("arch", [9, 10, 12])
def test_auto_cp_unchanged_on_sm90_sm100_sm120(monkeypatch, arch):
    """The regression this file exists for.

    One sequence at one head is the shape `should_use_cp_host` was written for:
    too little work for the fused kernel to fill the device. No caller passes a
    host-side maximum, so auto CP has to hold without one.
    """
    call = _inputs([4096], 1)
    got = _dispatch(monkeypatch, arch=arch, **call)
    assert got == {9: "cp_delta_rule_dsl_sm90", 10: "cp_delta_rule_dsl_sm100",
                   12: "cp_delta_rule_dsl_sm120"}[arch]


@pytest.mark.parametrize("arch", [9, 10, 12])
def test_auto_still_declines_cp_when_there_is_plenty_of_work(monkeypatch, arch):
    """The other half of the heuristic still applies: it is not always CP."""
    call = _inputs([512] * 32, 8)
    got = _dispatch(monkeypatch, arch=arch, sm_count=80, **call)
    assert got == {9: "chunk_gated_delta_rule_sm90",
                   10: "chunk_gated_delta_rule_sm100",
                   12: "chunk_gated_delta_rule_sm120"}[arch]


def test_auto_on_sm80_is_the_fused_kernel(monkeypatch):
    """SM80 auto CP stays off, at every shape, including the ones CP wins.

    `1x8192 h1` is measured at 2.6x with a prepared plan and roughly at parity
    without one. Auto dispatch is not wired until the competition gate in
    the measurement harness's `cp_crossover_table.py` passes.
    """
    for seq_lens, heads in (([4096], 1), ([8192], 1), ([16384], 1),
                            ([2048], 4), ([512] * 8, 8)):
        call = _inputs(seq_lens, heads)
        assert _dispatch(monkeypatch, arch=8, **call) == \
            "chunk_gated_delta_rule_sm80"


def test_explicit_cp_on_sm80_still_runs_cp(monkeypatch):
    """`use_cp=True` is the only way in on SM80, and it still works."""
    call = _inputs([4096], 1)
    assert _dispatch(monkeypatch, arch=8, use_cp=True, **call) == \
        "cp_delta_rule_dsl_sm80"


def test_explicit_cp_false_never_runs_cp(monkeypatch):
    for arch, fused in ((8, "chunk_gated_delta_rule_sm80"),
                        (9, "chunk_gated_delta_rule_sm90"),
                        (10, "chunk_gated_delta_rule_sm100"),
                        (12, "chunk_gated_delta_rule_sm120")):
        call = _inputs([4096], 1)
        assert _dispatch(monkeypatch, arch=arch, use_cp=False, **call) == fused


def test_the_private_maximum_does_not_change_dispatch(monkeypatch):
    """It sizes per-sequence indexing. It is not a routing hint.

    Whatever it says, auto picks the same entry it would have picked without
    it. If that stops being true, the rule has crept back into the wrapper.
    """
    for arch in (8, 9, 10, 12):
        base = _dispatch(monkeypatch, arch=arch, **_inputs([4096], 1))
        for mx in (None, 1, 4096):
            got = _dispatch(monkeypatch, arch=arch, _max_seq_len=mx,
                            **_inputs([4096], 1))
            assert got == base, f"arch {arch}, _max_seq_len={mx}"


@pytest.mark.parametrize("bad", [0, -1, 4097, 1.0, True, "4096"])
def test_the_private_maximum_is_validated(bad):
    """A value below the real maximum under-sizes indexing, so it is checked."""
    call = _inputs([4096], 1)
    with pytest.raises(ValueError):
        gp.chunk_gated_delta_rule(_max_seq_len=bad, **call)


def test_the_private_maximum_accepts_the_edges():
    """1 and `total_seq_len` are both legal; only the kernel entry is mocked."""
    for mx in (1, 4096):
        call = _inputs([4096], 1)
        out = gp.chunk_gated_delta_rule(_max_seq_len=mx, use_cp=False, **call)
        assert out.shape == (4096, 1, HEAD_SIZE)
