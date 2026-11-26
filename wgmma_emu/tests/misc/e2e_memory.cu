// End-to-end check of the sm_90 memory-instruction qualifiers rewritten by
// tools/ptxemu/rules_misc.py (fork-only). Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Covered: ld / st with .cluster scope (-> .gpu), ld.proxy::readonly (qualifier dropped),
// prefetch{.global}.L1::32B.valid_addr (-> prefetch{.global}.L1), isspacep.shared::cluster and
// cvta{.to}.shared::cluster (-> .shared, cluster of one CTA). Every program carries
// .target sm_90a, which the core retargets (inventory row \.target).
#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kN = 1 << 14, kThreads = 256;

// out[8 i + k]: k = 0 ld.relaxed.cluster.global, 1 ld.acquire.cluster.global,
// 2 ld.global.relaxed.cluster (other qualifier order), 3 ld.global.proxy::readonly,
// 4 ld.proxy::readonly (generic), 5 load after both prefetch forms; st0 / st1 / st2 receive
// st.relaxed.cluster.global, st.release.cluster.global, st.relaxed.cluster (generic).
__global__ void k_ldst(const uint32_t* in, uint32_t* out, uint32_t* st0, uint32_t* st1,
                       uint32_t* st2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t* p = in + i;
  uint32_t v[6];
  asm volatile("ld.relaxed.cluster.global.u32 %0, [%1];" : "=r"(v[0]) : "l"(p) : "memory");
  asm volatile("ld.acquire.cluster.global.u32 %0, [%1];" : "=r"(v[1]) : "l"(p) : "memory");
  asm volatile("ld.global.relaxed.cluster.u32 %0, [%1];" : "=r"(v[2]) : "l"(p) : "memory");
  asm volatile("ld.global.u32.proxy::readonly %0, [%1];" : "=r"(v[3]) : "l"(p));
  asm volatile("ld.u32.proxy::readonly %0, [%1];" : "=r"(v[4]) : "l"(p));
  asm volatile("prefetch.global.L1::32B.valid_addr [%0];" ::"l"(in + (i ^ 7)));
  asm volatile("prefetch.L1::32B.valid_addr [%0];" ::"l"(in + (i ^ 9)));
  v[5] = in[i ^ 7];
  for (int k = 0; k < 6; ++k) out[8 * i + k] = v[k];
  asm volatile("st.relaxed.cluster.global.u32 [%0], %1;" ::"l"(st0 + i), "r"(v[0] + 1u) : "memory");
  asm volatile("st.release.cluster.global.u32 [%0], %1;" ::"l"(st1 + i), "r"(v[0] + 2u) : "memory");
  asm volatile("st.relaxed.cluster.u32 [%0], %1;" ::"l"(st2 + i), "r"(v[0] + 3u) : "memory");
}

// out[4 t + k]: k = 0 isspacep.shared::cluster(shared), 1 (global), 2 (local);
// 3 value read back through cvta.shared::cluster / cvta.to.shared::cluster round trips.
__global__ void k_space(uint32_t* gbuf, uint32_t* out) {
  __shared__ uint32_t s[2 * kThreads];
  const int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
  volatile uint32_t loc[2];
  loc[0] = (uint32_t)i;
  loc[1] = 0;
  s[t] = 0;
  s[kThreads + t] = 0;
  __syncthreads();
  uint32_t r[3];
  asm volatile("{ .reg .pred p; isspacep.shared::cluster p, %1; selp.u32 %0, 1, 0, p; }"
               : "=r"(r[0])
               : "l"(&s[t]));
  asm volatile("{ .reg .pred p; isspacep.shared::cluster p, %1; selp.u32 %0, 1, 0, p; }"
               : "=r"(r[1])
               : "l"(gbuf + i));
  asm volatile("{ .reg .pred p; isspacep.shared::cluster p, %1; selp.u32 %0, 1, 0, p; }"
               : "=r"(r[2])
               : "l"((const uint32_t*)loc));
  // shared::cta address -> generic (cvta.shared::cluster) -> store through the generic pointer.
  const uint64_t a = (uint64_t)__cvta_generic_to_shared(&s[t]);
  uint64_t g, a2;
  asm volatile("cvta.shared::cluster.u64 %0, %1;" : "=l"(g) : "l"(a));
  *(volatile uint32_t*)g = 0xA5000000u + (uint32_t)i;
  // generic -> shared::cluster address (cvta.to.shared::cluster) -> st.shared::cluster.
  asm volatile("cvta.to.shared::cluster.u64 %0, %1;" : "=l"(a2) : "l"(&s[kThreads + t]));
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"l"(a2), "r"(0x5A000000u + (uint32_t)i)
               : "memory");
  __syncthreads();
  out[4 * i] = r[0];
  out[4 * i + 1] = r[1];
  out[4 * i + 2] = r[2];
  out[4 * i + 3] = (s[t] == 0xA5000000u + (uint32_t)i) |
                   ((s[kThreads + t] == 0x5A000000u + (uint32_t)i) << 1) |
                   (loc[0] == (uint32_t)i ? 4u : 0u);
}

int main() {
  {
    std::vector<uint32_t> in(kN);
    Rng r(0x3E3);
    for (uint32_t& x : in) x = r.u32();
    Dev<uint32_t> din(in), out(8 * kN), st0(kN), st1(kN), st2(kN);
    k_ldst<<<kN / kThreads, kThreads>>>(din.p, out.p, st0.p, st1.p, st2.p);
    MISC_SYNC();
    const std::vector<uint32_t> o = out.get(), s0 = st0.get(), s1 = st1.get(), s2 = st2.get();
    Item ic("misc.e2e.ld_st.cluster");
    Item ir("misc.e2e.ld.proxy::readonly");
    Item ip("misc.e2e.prefetch.L1::32B.valid_addr");
    for (int i = 0; i < kN; ++i) {
      for (int k = 0; k < 3; ++k)
        ic.check(o[8 * i + k] == in[i], "i=%d ld form=%d got=%08x want=%08x", i, k, o[8 * i + k],
                 in[i]);
      ic.check(s0[i] == in[i] + 1u && s1[i] == in[i] + 2u && s2[i] == in[i] + 3u,
               "i=%d st got=%08x,%08x,%08x base=%08x", i, s0[i], s1[i], s2[i], in[i]);
      for (int k = 3; k < 5; ++k)
        ir.check(o[8 * i + k] == in[i], "i=%d form=%d got=%08x want=%08x", i, k, o[8 * i + k],
                 in[i]);
      ip.check(o[8 * i + 5] == in[i ^ 7], "i=%d got=%08x want=%08x", i, o[8 * i + 5], in[i ^ 7]);
    }
  }
  {
    const int blocks = 8, n = blocks * kThreads;
    Dev<uint32_t> g(n), out(4 * n);
    k_space<<<blocks, kThreads>>>(g.p, out.p);
    MISC_SYNC();
    const std::vector<uint32_t> o = out.get();
    Item is("misc.e2e.isspacep.shared::cluster");
    Item ia("misc.e2e.cvta.shared::cluster");
    for (int i = 0; i < n; ++i) {
      is.check(o[4 * i] == 1u && o[4 * i + 1] == 0u && o[4 * i + 2] == 0u,
               "i=%d shared=%u global=%u local=%u", i, o[4 * i], o[4 * i + 1], o[4 * i + 2]);
      ia.check(o[4 * i + 3] == 7u, "i=%d flags=%u", i, o[4 * i + 3]);
    }
  }
  return finish();
}
