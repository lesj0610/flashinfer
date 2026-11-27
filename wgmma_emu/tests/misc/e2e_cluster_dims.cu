// End-to-end check of kernels with compile-time cluster directives (sm_90): __cluster_dims__
// emits .explicitcluster / .reqnctapercluster, a third __launch_bounds__ argument emits
// .maxclusterrank. The core (T1) strips them for a cluster of one CTA. Fork-only. Owner: T2.
// Built for sm_90a by tools/nvcc_emu.
//
// Needs Ctx.cluster_info (T1) for %is_explicit_cluster of the explicit kernel; kept in its own
// program so that the missing piece fails only this test.
// [inference] A kernel compiled with __cluster_dims__ is an explicit cluster launch even when
// launched with <<<>>> (ptx.txt 896-900). Without .explicitcluster / .reqnctapercluster the
// value depends on the launch attributes, which host/shim.c drops, so the translator rejects
// the read (tests/translator/test_misc_cluster_b128.py); k_maxrank does not read it.
#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kOut = 6;

template <bool kReadExplicit>
__device__ __forceinline__ void cluster_regs(uint32_t* o) {
  if constexpr (kReadExplicit)
    asm volatile("{ .reg .pred p; mov.pred p, %%is_explicit_cluster; selp.u32 %0, 1, 0, p; }"
                 : "=r"(o[0]));
  else
    o[0] = 0;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(o[1]));
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(o[2]));
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(o[3]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(o[4]));
}

__global__ void __cluster_dims__(1, 1, 1) k_explicit(uint32_t* out) {
  uint32_t* o = out + kOut * (blockIdx.x * blockDim.x + threadIdx.x);
  cluster_regs<true>(o);
  asm volatile("barrier.cluster.arrive;\n\tbarrier.cluster.wait;" ::: "memory");
  o[5] = blockIdx.x;
}

__global__ void __launch_bounds__(128, 1, 1) k_maxrank(uint32_t* out) {
  uint32_t* o = out + kOut * (blockIdx.x * blockDim.x + threadIdx.x);
  cluster_regs<false>(o);
  o[5] = blockIdx.x;
}

typedef void (*K)(uint32_t*);

static void run(const char* name, K k, uint32_t explicit_cluster) {
  const int blocks = 6, threads = 128;
  Dev<uint32_t> out((size_t)kOut * blocks * threads);
  k<<<blocks, threads>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it(name);
  for (int b = 0; b < blocks; ++b)
    for (int t = 0; t < threads; ++t) {
      const uint32_t* v = &o[kOut * (b * threads + t)];
      const uint32_t want[kOut] = {explicit_cluster, 1, 0, (uint32_t)b, 1, (uint32_t)b};
      for (int k2 = 0; k2 < kOut; ++k2)
        it.check(v[k2] == want[k2], "block=%d thread=%d field=%d got=%u want=%u", b, t, k2, v[k2],
                 want[k2]);
    }
}

int main() {
  run("misc.e2e.cluster_dims.explicit", k_explicit, 1);
  run("misc.e2e.cluster_dims.maxclusterrank", k_maxrank, 0);
  return finish();
}
