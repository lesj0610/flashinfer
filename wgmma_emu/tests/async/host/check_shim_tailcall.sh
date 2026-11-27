#!/usr/bin/env bash
# The shim's dlsym must reach glibc's dlsym through a tail jump for the names it does not
# intercept, so the return address glibc uses to resolve RTLD_NEXT stays the caller's
# (host/shim.c, dlsym). Checks the disassembly of the exported dlsym for an indirect jmp.
#   tests/async/host/check_shim_tailcall.sh build/libwgemu_shim.so
set -euo pipefail
so=${1:?usage: check_shim_tailcall.sh <libwgemu_shim.so>}
body=$(objdump -d --no-show-raw-insn "$so" | awk '/^[0-9a-f]+ <dlsym>:$/ { f = 1; next } f && /^$/ { exit } f')
if [ -z "$body" ]; then
  echo "check_shim_tailcall: no dlsym symbol in $so"
  echo "WGEMU-TEST name=host.shim.dlsym_tail_call cases=1 mismatches=1"
  exit 1
fi
if grep -Eq 'jmp +\*' <<<"$body"; then
  echo "WGEMU-TEST name=host.shim.dlsym_tail_call cases=1 mismatches=0"
else
  echo "$body"
  echo "WGEMU-TEST name=host.shim.dlsym_tail_call cases=1 mismatches=1"
  exit 1
fi
