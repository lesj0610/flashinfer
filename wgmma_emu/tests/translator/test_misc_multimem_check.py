"""check mode: multimem.cp.async.bulk, delegated by rules_misc to rules_async, must get the
bulk-state register initialized by rules_async.function_prologue (CPU only)."""

from __future__ import annotations

import re

from ptxemu import rules_async, rules_misc
from ptxemu.core import translate

SRC = (
    ".version 8.0\n.target sm_90a\n.address_size 64\n\n"
    ".visible .entry k(\n\t.param .u64 k_param_0\n)\n{\n"
    "\t.reg .b32 \t%r<2>;\n\t.reg .b64 \t%rd<3>;\n"
    "\t.shared .align 16 .b8 buf[64];\n"
    "\tld.param.u64 \t%rd1, [k_param_0];\n\tmov.u32 \t%r1, buf;\n"
    "\tmultimem.cp.async.bulk.global.shared::cta.bulk_group \t[%rd1], [%r1], 64;\n"
    "\tcp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;\n\tret;\n}\n"
)


def test_multimem_bulk_state_initialized():
    out = translate(
        SRC, mode="check", rules=[rules_async, rules_misc], warn_stream=None
    )
    m = re.search(r"cvta\.local\.u64\s+(%\w+),", out)
    assert m, "bulk state is never initialized"
    reg = m.group(1)
    init = out.index("wgemu_bulk_state_init,")
    s2g = out.index("wgemu_bulk_s2g,")
    assert m.start() < init < s2g
    # the register handed to the bulk copy is the initialized one
    assert re.search(
        rf"st\.param\.b64\s+\[param4\+0\], {re.escape(reg)};", out[init:s2g]
    )
