// End-to-end check of statically allocated shared memory above 48 KB per CTA, which sm_90a
// allows (ptx.txt 2280-2290) and sm_80 does not. Fork-only. Owner: T2. Built for sm_90a by
// tools/nvcc_emu.
//
// WGEMU-COVERS-FEATURE: static_smem_over_48k
//
// Not in e2e_manifest.json: phase 2 (integrator 11-25). The core does not move such
// variables yet and ptxas -arch=sm_80 rejects the kernel explicitly.
// Needs the core (T1) to move such .shared variables into dynamic shared memory; kept in its
// own program so that the missing piece fails only this test.
// [inference] The core places the moved variables at offset 0 of the dynamic shared memory of a
// kernel that declares none of its own, so the host opts in to and launches with the static size
// as dynamic bytes; on sm_90a the unused dynamic bytes are harmless (static + dynamic = 128 KB,
// below the opt-in limit).
#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kThreads = 256;
constexpr int kBigWords = 16384;  // 64 KB in one variable
constexpr int kAWords = 10240;    // 40 KB
constexpr int kBWords = 6144;     // 24 KB, 64 KB together with the 40 KB array

__device__ __host__ inline uint32_t pattern(int b, int i) {
  return 0x9E3779B9u * (uint32_t)(i + 1) ^ (uint32_t)b;
}

// One 64 KB array: write all of it, read it back reversed after a barrier.
__global__ void k_big(uint32_t* out) {
  __shared__ __align__(16) uint32_t big[kBigWords];
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < kBigWords; i += kThreads) big[i] = pattern(b, i);
  __syncthreads();
  for (int i = threadIdx.x; i < kBigWords; i += kThreads)
    out[b * kBigWords + i] = big[kBigWords - 1 - i];
}

// Two arrays below 48 KB each, 64 KB together.
__global__ void k_two(uint32_t* out) {
  __shared__ __align__(16) uint32_t a[kAWords];
  __shared__ __align__(16) uint32_t c[kBWords];
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < kAWords; i += kThreads) a[i] = pattern(b, i);
  for (int i = threadIdx.x; i < kBWords; i += kThreads) c[i] = pattern(b, kAWords + i);
  __syncthreads();
  for (int i = threadIdx.x; i < kAWords + kBWords; i += kThreads) {
    const int j = kAWords + kBWords - 1 - i;
    out[b * (kAWords + kBWords) + i] = j < kAWords ? a[j] : c[j - kAWords];
  }
}

template <class K>
static void run(const char* name, K k, int words) {
  const int blocks = 4;
  MISC_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, words * 4));
  Dev<uint32_t> out((size_t)blocks * words);
  k<<<blocks, kThreads, words * 4>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it(name);
  for (int b = 0; b < blocks; ++b)
    for (int i = 0; i < words; ++i) {
      const uint32_t want = pattern(b, words - 1 - i);
      it.check(o[(size_t)b * words + i] == want, "block=%d i=%d got=%08x want=%08x", b, i,
               o[(size_t)b * words + i], want);
    }
}

int main() {
  run("misc.e2e.static_smem.64k_one", k_big, kBigWords);
  run("misc.e2e.static_smem.64k_two", k_two, kAWords + kBWords);
  return finish();
}
