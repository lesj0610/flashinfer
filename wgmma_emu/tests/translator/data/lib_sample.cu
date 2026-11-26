// Library-shaped sample for translator merge tests (PTX only, compute_80 -rdc).
#include <cstdint>
#include <cstdio>

static __device__ __noinline__ uint32_t lib_inner(uint32_t a) {
  printf("WGEMU-LIB inner %u\n", a);
  return a * 3u;
}

template <int N>
__device__ __noinline__ uint32_t lib_tmpl(uint32_t a) {
  return a + N;
}

extern "C" __device__ __noinline__ uint32_t wgemu_sample_u32(uint32_t x, uint64_t p) {
  return lib_inner(x) + static_cast<uint32_t>(p) + lib_tmpl<3>(x);
}

extern "C" __device__ __noinline__ void wgemu_sample_void(uint64_t p) {
  *reinterpret_cast<uint32_t*>(p) = 1u;
}

extern "C" __device__ __noinline__ void wgemu_sample_unused(uint64_t p) {
  printf("WGEMU-LIB unused\n");
  *reinterpret_cast<uint32_t*>(p) = 2u;
}
