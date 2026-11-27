#!/usr/bin/env bash
# Host-side tests of host/shim.c without nvcc_emu: builds the shim, two helper libraries and
# test_shim_driver.c, then runs
#   check_shim_tailcall.sh                         host.shim.dlsym_tail_call
#   test_shim_driver        (LD_PRELOAD shim)      host.shim.*, the device's own compute capability
#   test_shim_driver spoof  (+ WGEMU_SPOOF_CC=1)   host.shim.*, compute capability 9.0 reported
# Needs gcc, objdump, cuda.h (CUDA_HOME, default /usr/local/cuda), libcuda and one GPU, picked
# with CUDA_VISIBLE_DEVICES. Build directory: WGEMU_HOST_BUILD (default <root>/build/async_host).
# Exit status 0 only when every WGEMU-TEST line reports mismatches=0.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda}
out=${WGEMU_HOST_BUILD:-$root/build/async_host}
mkdir -p "$out"

gcc -O2 -shared -fPIC "$root/host/shim.c" -ldl -o "$out/libwgemu_shim.so"
gcc -O2 -shared -fPIC "$here/next_a.c" -ldl -o "$out/libnext_a.so"
gcc -O2 -shared -fPIC "$here/next_b.c" -o "$out/libnext_b.so"
# --no-as-needed keeps libnext_b in DT_NEEDED (nothing references it directly) right after
# libnext_a, which is what RTLD_NEXT from libnext_a must find.
gcc -O2 -std=c11 -Wall -I"$cuda/include" -I"$root/include" "$here/test_shim_driver.c" \
  -L"$out" -Wl,--no-as-needed -lnext_a -lnext_b -Wl,--as-needed -Wl,-rpath,"$out" \
  -lcuda -ldl -o "$out/test_shim_driver"

status=0
"$here/check_shim_tailcall.sh" "$out/libwgemu_shim.so" || status=1
LD_PRELOAD="$out/libwgemu_shim.so" "$out/test_shim_driver" || status=1
WGEMU_SPOOF_CC=1 LD_PRELOAD="$out/libwgemu_shim.so" "$out/test_shim_driver" spoof || status=1
exit "$status"
