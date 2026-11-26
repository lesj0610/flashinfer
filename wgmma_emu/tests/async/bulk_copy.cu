// Non-tensor bulk copy / reduction end-to-end test (ptx.txt 15300-15780).
//
// One CTA of 128 threads; thread 0 issues
//   global -> shared  : plain, .L2::cache_hint, .multicast::cluster (mask = own CTA), .shared::cta
//                       destination; completion through one mbarrier with expect_tx;
//   L2 prefetch       : cp.async.bulk.prefetch.L2 (with and without cache hint);
//   shared -> global  : copy (plain, cache hint) and reductions add.u32 / min.s32 / xor.b32 /
//                       add.f32 / max.u32 (cache hint) in one bulk group, wait_group.read then
//                       wait_group;
//   shared -> shared  : cp.reduce.async.bulk .shared::cluster.shared::cta add.u32 / and.b32 into
//                       the CTA's own shared memory (cluster of one), mbarrier completion.
// The host checks every byte against a CPU reference.
#include "async_common.cuh"

constexpr int kThreads = 128;
constexpr int kG2S = 2048;  // bytes copied global -> shared
constexpr int kS2G = 2560;  // bytes copied shared -> global
constexpr int kRedN = 64;   // elements per reduction array
constexpr int kRedArrays = 5;

// Shared-memory map (bytes).
constexpr int kOffG2S = 0;
constexpr int kOffS2G = kOffG2S + kG2S;
constexpr int kOffRed = kOffS2G + kS2G;
constexpr int kOffSrcA = kOffRed + kRedArrays * kRedN * 4;
constexpr int kOffDstA = kOffSrcA + kRedN * 4;
constexpr int kOffDstB = kOffDstA + kRedN * 4;
constexpr int kSmemBytes = kOffDstB + kRedN * 4;

struct Params {
  const uint8_t* gsrc;  // kG2S bytes
  uint8_t* gdst;        // kS2G bytes
  uint32_t* gred;       // kRedArrays * kRedN words, pre-initialized
  uint32_t* out_g2s;    // kG2S / 4 words
  uint32_t* out_s2s;    // 2 * kRedN words
};

__host__ __device__ inline uint32_t s2g_word(uint32_t i) { return 0xA5000000u ^ wg_hash32(i); }
__host__ __device__ inline uint32_t red_src(int a, uint32_t i) {
  switch (a) {
    case 0:
      return i * 3u + 1u;
    case 1:
      return (uint32_t)((int32_t)(i * 7u) - 200);
    case 2:
      return wg_hash32(i + 1000u);
    case 3: {
      const float f = (float)i - 20.0f;
      uint32_t u;
      memcpy(&u, &f, 4);
      return u;
    }
    default:
      return wg_hash32(i + 3000u);
  }
}
__host__ __device__ inline uint32_t dst_b_init(uint32_t i) { return 0xF0F0F0F0u ^ wg_hash32(i); }

__global__ void k_bulk(Params p) {
  __shared__ alignas(128) uint8_t sm[kSmemBytes];
  __shared__ alignas(8) uint64_t bars[2];
  const uint32_t t = threadIdx.x;
  const uint32_t base = smem_u32(sm);
  const uint32_t bar0 = smem_u32(&bars[0]);
  const uint32_t bar1 = smem_u32(&bars[1]);
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  if (t == 0) {
    mbar_init(bar0, 1);
    mbar_init(bar1, 1);
    fence_mbar_init();
  }
  // Generic-proxy writes of every source / destination that the async proxy reads.
  for (uint32_t i = t; i < kS2G / 4; i += kThreads) w[kOffS2G / 4 + i] = s2g_word(i);
  for (uint32_t i = t; i < kRedArrays * kRedN; i += kThreads) {
    w[kOffRed / 4 + i] = red_src((int)(i / kRedN), i % kRedN);
  }
  for (uint32_t i = t; i < kRedN; i += kThreads) {
    w[kOffSrcA / 4 + i] = i + 1u;
    w[kOffDstA / 4 + i] = 1000u * i;
    w[kOffDstB / 4 + i] = dst_b_init(i);
  }
  fence_proxy_async_smem();
  __syncthreads();

  if (t == 0) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const uint16_t mask = 1;
    mbar_arrive_expect_tx(bar0, kG2S);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
            "r"(base + kOffG2S),
        "l"(p.gsrc), "r"(1024), "r"(bar0)
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1], 512, [%2], %3;" ::"r"(base + kOffG2S + 1024),
        "l"(p.gsrc + 1024), "r"(bar0), "l"(pol)
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1], 256, [%2], %3;" ::"r"(base + kOffG2S + 1536),
        "l"(p.gsrc + 1536), "r"(bar0), "h"(mask)
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" ::
            "r"(base + kOffG2S + 1792),
        "l"(p.gsrc + 1792), "r"(bar0)
        : "memory");
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 1024;" ::"l"(p.gsrc) : "memory");
    asm volatile(
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], 512, %1;" ::"l"(p.gsrc + 1024),
        "l"(pol)
        : "memory");

    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 2048;" ::"l"(p.gdst),
                 "r"(base + kOffS2G)
                 : "memory");
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], 512, %2;" ::"l"(
            p.gdst + 2048),
        "r"(base + kOffS2G + 2048), "l"(pol)
        : "memory");
    const uint32_t rs = base + kOffRed;
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], 256;" ::"l"(p.gred),
        "r"(rs)
        : "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0], [%1], 256;" ::"l"(p.gred +
                                                                                            kRedN),
        "r"(rs + 256)
        : "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], 256;" ::"l"(
            p.gred + 2 * kRedN),
        "r"(rs + 512)
        : "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 256;" ::"l"(
            p.gred + 3 * kRedN),
        "r"(rs + 768)
        : "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.max.u32"
        " [%0], [%1], 256, %2;" ::"l"(p.gred + 4 * kRedN),
        "r"(rs + 1024), "l"(pol)
        : "memory");
    bulk_commit();
    bulk_wait_read0();
    bulk_wait0();

    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar1),
                 "r"(2 * kRedN * 4)
                 : "memory");
    asm volatile(
        "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32"
        " [%0], [%1], 256, [%2];" ::"r"(base + kOffDstA),
        "r"(base + kOffSrcA), "r"(bar1)
        : "memory");
    asm volatile(
        "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32"
        " [%0], [%1], 256, [%2];" ::"r"(base + kOffDstB),
        "r"(base + kOffSrcA), "r"(bar1)
        : "memory");
  }
  mbar_wait(bar0, 0);
  mbar_wait(bar1, 0);
  for (uint32_t i = t; i < kG2S / 4; i += kThreads) p.out_g2s[i] = w[kOffG2S / 4 + i];
  for (uint32_t i = t; i < kRedN; i += kThreads) {
    p.out_s2s[i] = w[kOffDstA / 4 + i];
    p.out_s2s[kRedN + i] = w[kOffDstB / 4 + i];
  }
}

static uint32_t red_init(int a, uint32_t i) {
  switch (a) {
    case 0:
      return wg_hash32(i + 5000u);
    case 1:
      return (uint32_t)((int32_t)(wg_hash32(i + 6000u) % 2000u) - 1000);
    case 2:
      return wg_hash32(i + 7000u);
    case 3: {
      const float f = (float)(i % 7u) * 0.5f;
      uint32_t u;
      memcpy(&u, &f, 4);
      return u;
    }
    default:
      return wg_hash32(i + 8000u);
  }
}

static uint32_t red_ref(int a, uint32_t i) {
  const uint32_t x = red_init(a, i), v = red_src(a, i);
  switch (a) {
    case 0:
      return x + v;
    case 1:
      return (int32_t)x < (int32_t)v ? x : v;
    case 2:
      return x ^ v;
    case 3: {
      float fx, fv;
      memcpy(&fx, &x, 4);
      memcpy(&fv, &v, 4);
      const float s = fx + fv;  // small integers and halves: exact in f32
      uint32_t u;
      memcpy(&u, &s, 4);
      return u;
    }
    default:
      return x > v ? x : v;
  }
}

int main() {
  std::vector<uint32_t> src(kG2S / 4), red(kRedArrays * kRedN);
  for (uint32_t i = 0; i < kG2S / 4; ++i) src[i] = wg_hash32(i + 77u);
  for (int a = 0; a < kRedArrays; ++a) {
    for (uint32_t i = 0; i < kRedN; ++i) red[a * kRedN + i] = red_init(a, i);
  }
  Params p;
  uint8_t *d_src, *d_dst;
  uint32_t *d_red, *d_g2s, *d_s2s;
  WG_CHECK(cudaMalloc(&d_src, kG2S));
  WG_CHECK(cudaMalloc(&d_dst, kS2G));
  WG_CHECK(cudaMalloc(&d_red, 4 * kRedArrays * kRedN));
  WG_CHECK(cudaMalloc(&d_g2s, kG2S));
  WG_CHECK(cudaMalloc(&d_s2s, 4 * 2 * kRedN));
  WG_CHECK(cudaMemcpy(d_src, src.data(), kG2S, cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemcpy(d_red, red.data(), 4 * kRedArrays * kRedN, cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_dst, 0, kS2G));
  WG_CHECK(cudaMemset(d_g2s, 0, kG2S));
  WG_CHECK(cudaMemset(d_s2s, 0, 4 * 2 * kRedN));
  p.gsrc = d_src;
  p.gdst = d_dst;
  p.gred = d_red;
  p.out_g2s = d_g2s;
  p.out_s2s = d_s2s;
  k_bulk<<<1, kThreads>>>(p);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> g2s(kG2S / 4), dst(kS2G / 4), ro(kRedArrays * kRedN), s2s(2 * kRedN);
  WG_CHECK(cudaMemcpy(g2s.data(), d_g2s, kG2S, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(dst.data(), d_dst, kS2G, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(ro.data(), d_red, 4 * kRedArrays * kRedN, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(s2s.data(), d_s2s, 4 * 2 * kRedN, cudaMemcpyDeviceToHost));

  Checker c1("async.bulk.global_to_shared");
  for (uint32_t i = 0; i < kG2S / 4; ++i) c1.eq(i, g2s[i], src[i]);
  wg_report(c1);

  Checker c2("async.bulk.shared_to_global");
  for (uint32_t i = 0; i < kS2G / 4; ++i) c2.eq(i, dst[i], s2g_word(i));
  wg_report(c2);

  static const char* rn[kRedArrays] = {
      "async.bulk.reduce_global.add_u32", "async.bulk.reduce_global.min_s32",
      "async.bulk.reduce_global.xor_b32", "async.bulk.reduce_global.add_f32",
      "async.bulk.reduce_global.max_u32_cache_hint"};
  for (int a = 0; a < kRedArrays; ++a) {
    Checker c(rn[a]);
    for (uint32_t i = 0; i < kRedN; ++i) c.eq(i, ro[a * kRedN + i], red_ref(a, i));
    wg_report(c);
  }

  Checker c3("async.bulk.reduce_shared");
  for (uint32_t i = 0; i < kRedN; ++i) {
    c3.eq(i, s2s[i], 1000u * i + i + 1u);
    c3.eq(kRedN + i, s2s[kRedN + i], dst_b_init(i) & (i + 1u));
  }
  wg_report(c3);

  WG_CHECK(cudaFree(d_src));
  WG_CHECK(cudaFree(d_dst));
  WG_CHECK(cudaFree(d_red));
  WG_CHECK(cudaFree(d_g2s));
  WG_CHECK(cudaFree(d_s2s));
  return wg_finish();
}
