#include <cstdint>
extern "C" __device__ __noinline__ void wgemu_b(uint64_t p);
extern "C" __device__ __noinline__ void wgemu_f(uint32_t a, uint64_t p);
extern "C" __device__ __noinline__ uint64_t wgemu_d(uint32_t x);
__global__ void kc(uint64_t* p) { wgemu_b((uint64_t)p); wgemu_f(threadIdx.x, 5); p[0] = wgemu_d(threadIdx.x); }
