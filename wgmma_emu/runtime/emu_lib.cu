// Emulation library translation unit (PLAN section 6). Compiled once to emu_lib.ptx by
// tools/nvcc_emu:
//   nvcc -arch=compute_80 -rdc=true --ptx -O3 -std=c++17 -I include runtime/emu_lib.cu
// The translator merges the pruned library PTX into each translated module. The three parts
// define disjoint wgemu_* entry points.
#include "lib_wgmma.cu"
#include "lib_async.cu"
#include "lib_misc.cu"
