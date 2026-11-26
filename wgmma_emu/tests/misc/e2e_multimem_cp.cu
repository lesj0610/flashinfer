// End-to-end check of multimem.cp.async.bulk / multimem.cp.reduce.async.bulk (sm_90), which
// tools/ptxemu/rules_misc.py turns into the plain bulk forms and hands to rules_async (A).
// Fork-only. Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Depends on rules_async (cp.async.bulk / cp.reduce.async.bulk with .bulk_group,
// fence.proxy.async.shared::cta, cp.async.bulk.commit_group / wait_group), so it is its own
// program. [inference] One replica: the multimem destination is an ordinary global address.
//
// Every block fills a shared source tile, then one thread issues: a copy into the block's own
// destination tile, and reductions of the same tile into accumulators shared by all blocks
// (add.u32, relaxed.gpu add.f32, add.noftz.bf16, min.s32). All values are small integers, so
// every sum is exact in any order.
#include <algorithm>

#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kBlocks = 8, kThreads = 128, kWords = 256;  // 1 KB tile
constexpr uint32_t kBytes = kWords * 4;

__device__ __forceinline__ uint32_t src_u32(int b, int i) { return 1000u * b + (uint32_t)i; }
__device__ __host__ __forceinline__ int src_s32(int b, int i) { return (i % 17) - 3 * b; }

__global__ void k_mmcp(uint32_t* copy, uint32_t* acc_u32, float* acc_f32, uint16_t* acc_bf16,
                       int* acc_min) {
  __shared__ __align__(128) uint32_t su[kWords];
  __shared__ __align__(128) float sf[kWords];
  __shared__ __align__(128) uint16_t sb[2 * kWords];
  __shared__ __align__(128) int ss[kWords];
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < kWords; i += kThreads) {
    su[i] = src_u32(b, i);
    sf[i] = (float)(b + 1) * 0.5f + (float)i;
    ss[i] = src_s32(b, i);
  }
  for (int i = threadIdx.x; i < 2 * kWords; i += kThreads) {
    // bf16 of the small integer (i % 9) + 1: exact, and sums over 8 blocks stay below 256.
    const float f = (float)(i % 9 + 1);
    sb[i] = (uint16_t)(__float_as_uint(f) >> 16);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t a_su = (uint32_t)__cvta_generic_to_shared(su);
    const uint32_t a_sf = (uint32_t)__cvta_generic_to_shared(sf);
    const uint32_t a_sb = (uint32_t)__cvta_generic_to_shared(sb);
    const uint32_t a_ss = (uint32_t)__cvta_generic_to_shared(ss);
    asm volatile("multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                     copy + b * kWords),
                 "r"(a_su), "r"(kBytes)
                 : "memory");
    asm volatile(
        "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(
            acc_u32),
        "r"(a_su), "r"(kBytes)
        : "memory");
    asm volatile(
        "multimem.cp.reduce.async.bulk.relaxed.gpu.global.shared::cta.bulk_group.add.f32 "
        "[%0], [%1], %2;" ::"l"(acc_f32),
        "r"(a_sf), "r"(kBytes)
        : "memory");
    asm volatile(
        "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[%0], [%1], %2;" ::"l"(acc_bf16),
        "r"(a_sb), "r"(kBytes)
        : "memory");
    asm volatile(
        "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0], [%1], %2;" ::"l"(
            acc_min),
        "r"(a_ss), "r"(kBytes)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  __syncthreads();  // the shared sources stay alive until the bulk group completed
}

static float bf16_to_f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main() {
  Dev<uint32_t> copy(kBlocks * kWords), acc_u32(kWords);
  Dev<float> acc_f32(kWords);
  Dev<uint16_t> acc_bf16(2 * kWords);
  std::vector<int> min_init(kWords, 1 << 20);
  Dev<int> acc_min(min_init);
  k_mmcp<<<kBlocks, kThreads>>>(copy.p, acc_u32.p, acc_f32.p, acc_bf16.p, acc_min.p);
  MISC_SYNC();
  const std::vector<uint32_t> cv = copy.get(), au = acc_u32.get();
  const std::vector<float> af = acc_f32.get();
  const std::vector<uint16_t> ab = acc_bf16.get();
  const std::vector<int> am = acc_min.get();
  {
    Item it("misc.e2e.multimem.cp.async.bulk");
    for (int b = 0; b < kBlocks; ++b)
      for (int i = 0; i < kWords; ++i) {
        const uint32_t want = 1000u * b + (uint32_t)i;
        it.check(cv[b * kWords + i] == want, "block=%d i=%d got=%u want=%u", b, i,
                 cv[b * kWords + i], want);
      }
  }
  {
    Item it("misc.e2e.multimem.cp.reduce.async.bulk");
    for (int i = 0; i < kWords; ++i) {
      uint32_t wu = 0;
      float wf = 0;
      int wm = 1 << 20;
      for (int b = 0; b < kBlocks; ++b) {
        wu += 1000u * b + (uint32_t)i;
        wf += (float)(b + 1) * 0.5f + (float)i;
        wm = std::min(wm, src_s32(b, i));
      }
      it.check(au[i] == wu, "add.u32 i=%d got=%u want=%u", i, au[i], wu);
      it.check(af[i] == wf, "add.f32 i=%d got=%g want=%g", i, af[i], wf);
      it.check(am[i] == wm, "min.s32 i=%d got=%d want=%d", i, am[i], wm);
    }
    for (int i = 0; i < 2 * kWords; ++i) {
      const float want = (float)(kBlocks * (i % 9 + 1));
      it.check(bf16_to_f(ab[i]) == want, "add.noftz.bf16 i=%d got=%04x want=%g", i, ab[i], want);
    }
  }
  return finish();
}
