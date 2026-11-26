// Fence forms of sm_90 (ptx.txt 19619-19810) executed in data-carrying patterns.
//
// k_proxy (128 threads): generic-proxy writes handed to the async proxy and back.
//   A: shared writes, fence.proxy.async.shared::cta (+ fence.proxy.async), bulk store.
//   B: global writes, fence.proxy.async.global (+ membar.proxy.async.global), bulk load.
//   C: shared writes, fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster, bulk
//      store.
//   D: shared reads, then
//      fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster,
//      bulk load overwriting the read buffer (write-after-read), then
//      fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster and reads.
// k_mp (64 threads): warp 0 publishes data, warp 1 consumes it, once per release / acquire fence
//   pair (thread fences with .cluster / .gpu / .sys scope, .acquire / .release, alias proxy fences,
//   fence.alias, bidirectional proxy fences, sync_restrict fences around an mbarrier).
// The emulation maps these fences to native fences of at least the same strength; the test
// checks that every form translates and runs and that the data flows are intact.
#include "async_common.cuh"

constexpr int kThreads = 128;
constexpr int kRounds = 8;

__device__ __forceinline__ void bulk_s2g(void* g, const void* s, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(g),
               "r"(smem_u32(s)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* s, const void* g, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(s)),
      "l"(g), "r"(bytes), "r"(bar)
      : "memory");
}

struct ProxyArgs {
  uint32_t* out_a;       // 128
  uint32_t* gbuf;        // 128, written by the kernel
  uint32_t* out_b;       // 128
  uint32_t* out_c;       // 128
  const uint32_t* gsrc;  // 128, host data
  uint32_t* out_d;       // 128 (values read before the overwrite)
  uint32_t* out_e;       // 128 (values after the overwrite)
};

__host__ __device__ inline uint32_t pa(uint32_t i) { return 0xA0000000u ^ wg_hash32(i); }
__host__ __device__ inline uint32_t pb(uint32_t i) { return 0xB0000000u ^ wg_hash32(i + 256u); }
__host__ __device__ inline uint32_t pc(uint32_t i) { return 0xC0000000u ^ wg_hash32(i + 512u); }
__host__ __device__ inline uint32_t psrc(uint32_t i) { return 0xD0000000u ^ wg_hash32(i + 768u); }

__global__ void k_proxy(ProxyArgs a) {
  __shared__ alignas(128) uint32_t sa[kThreads], sb[kThreads], sc[kThreads];
  __shared__ alignas(8) uint64_t bar[2];
  const uint32_t t = threadIdx.x;
  const uint32_t b0 = smem_u32(&bar[0]), b1 = smem_u32(&bar[1]);
  if (t == 0) {
    mbar_init(b0, 1);
    mbar_init(b1, 1);
    fence_mbar_init();
  }
  // A.
  sa[t] = pa(t);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("fence.proxy.async;" ::: "memory");
    bulk_s2g(a.out_a, sa, kThreads * 4);
    bulk_commit();
  }
  // B.
  a.gbuf[t] = pb(t);
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("membar.proxy.async.global;" ::: "memory");
    mbar_arrive_expect_tx(b0, kThreads * 4);
    bulk_g2s(sb, a.gbuf, kThreads * 4, b0);
  }
  mbar_wait(b0, 0);
  a.out_b[t] = sb[(t + 1) % kThreads];
  // C.
  sc[t] = pc(t);
  asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                   : "memory");
  __syncthreads();
  if (t == 0) {
    bulk_s2g(a.out_c, sc, kThreads * 4);
    bulk_commit();
    bulk_wait_read0();  // A and C sources may be reused from here on
  }
  __syncthreads();
  // D: read sa, then overwrite it through the async proxy.
  const uint32_t before = sa[(t + 9) % kThreads];
  asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;" ::
                   : "memory");
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b1, kThreads * 4);
    bulk_g2s(sa, a.gsrc, kThreads * 4, b1);
  }
  mbar_wait(b1, 0);
  asm volatile("fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" ::
                   : "memory");
  a.out_d[t] = before;
  a.out_e[t] = sa[(t + 17) % kThreads];
  if (t == 0) bulk_wait0();
}

__device__ __forceinline__ void release_fence(int r) {
  switch (r) {
    case 0:
      asm volatile("fence.sc.cluster;" ::: "memory");
      break;
    case 1:
      asm volatile("fence.acq_rel.cluster;" ::: "memory");
      break;
    case 2:
      asm volatile("fence.release.gpu;" ::: "memory");
      break;
    case 3:
      asm volatile("fence.release.cluster;" ::: "memory");
      break;
    case 5:
      asm volatile("fence.proxy.alias.release.gpu;" ::: "memory");
      break;
    case 6:
      asm volatile("fence.alias.sync_restrict::read.release.sys;" ::: "memory");
      asm volatile("fence.proxy.alias.release.sys;" ::: "memory");
      break;
    default:
      asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
      asm volatile("membar.proxy.async;" ::: "memory");
      asm volatile("fence.sc.gpu;" ::: "memory");
      break;
  }
}

__device__ __forceinline__ void acquire_fence(int r) {
  switch (r) {
    case 0:
      asm volatile("fence.sc.cluster;" ::: "memory");
      break;
    case 1:
      asm volatile("fence.acq_rel.cluster;" ::: "memory");
      break;
    case 2:
      asm volatile("fence.acquire.gpu;" ::: "memory");
      break;
    case 3:
      asm volatile("fence.acquire.cluster;" ::: "memory");
      break;
    case 5:
      asm volatile("fence.proxy.alias.acquire.gpu;" ::: "memory");
      break;
    case 6:
      asm volatile("fence.proxy.alias.acquire.sys;" ::: "memory");
      break;
    default:
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("fence.sc.gpu;" ::: "memory");
      break;
  }
}

__host__ __device__ inline uint32_t mp_val(int r, uint32_t lane) {
  return (uint32_t)r * 1000u + lane * 7u + 1u;
}

// Round 4 publishes through an mbarrier with the sync_restrict fence pair of the ptx.txt
// 19780-19790 example; the other rounds through a volatile shared flag.
__global__ void k_mp(uint32_t* out) {
  __shared__ uint32_t data[kRounds][32];
  __shared__ volatile uint32_t flag;
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x, warp = t / 32, lane = t % 32;
  const uint32_t b = smem_u32(&bar);
  if (t == 0) {
    flag = 0;
    mbar_init(b, 32);
    fence_mbar_init();
  }
  __syncthreads();
  for (int r = 0; r < kRounds; ++r) {
    if (warp == 0) {
      data[r][lane] = mp_val(r, lane);
      if (r == 4) {
        uint64_t st;
        asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1];"
                     : "=l"(st)
                     : "r"(b)
                     : "memory");
      } else {
        __syncwarp();
        if (lane == 0) {
          release_fence(r);
          flag = (uint32_t)r + 1u;
        }
      }
    } else {
      if (r == 4) {
        uint32_t ok = 0;
        while (!ok) {
          asm volatile(
              "{\n\t.reg .pred p;\n\t"
              "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], 0;\n\t"
              "selp.u32 %0, 1, 0, p;\n\t}"
              : "=r"(ok)
              : "r"(b)
              : "memory");
        }
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
      } else {
        while (flag < (uint32_t)r + 1u) {
        }
        acquire_fence(r);
      }
      out[r * 32 + lane] = data[r][(lane + 3) % 32];
    }
  }
}

int main() {
  uint32_t* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 7 * kThreads * 4));
  WG_CHECK(cudaMemset(d, 0, 7 * kThreads * 4));
  std::vector<uint32_t> src(kThreads);
  for (int i = 0; i < kThreads; ++i) src[i] = psrc(i);
  ProxyArgs a{d,
              d + kThreads,
              d + 2 * kThreads,
              d + 3 * kThreads,
              d + 4 * kThreads,
              d + 5 * kThreads,
              d + 6 * kThreads};
  WG_CHECK(cudaMemcpy(d + 4 * kThreads, src.data(), kThreads * 4, cudaMemcpyHostToDevice));
  k_proxy<<<1, kThreads>>>(a);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> h(7 * kThreads);
  WG_CHECK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
  {
    Checker ck("async.fence.proxy_async_shared_to_bulk_store");
    for (int i = 0; i < kThreads; ++i) ck.eq(i, h[i], pa(i));
    wg_report(ck);
  }
  {
    Checker ck("async.fence.proxy_async_global_to_bulk_load");
    for (int i = 0; i < kThreads; ++i) ck.eq(i, h[2 * kThreads + i], pb((i + 1) % kThreads));
    wg_report(ck);
  }
  {
    Checker ck("async.fence.async_generic_release_shared_cta");
    for (int i = 0; i < kThreads; ++i) ck.eq(i, h[3 * kThreads + i], pc(i));
    wg_report(ck);
  }
  {
    Checker ck("async.fence.async_generic_read_release_and_acquire");
    for (int i = 0; i < kThreads; ++i) {
      ck.eq(2 * i, h[5 * kThreads + i], pa((i + 9) % kThreads));
      ck.eq(2 * i + 1, h[6 * kThreads + i], psrc((i + 17) % kThreads));
    }
    wg_report(ck);
  }

  uint32_t* dm = nullptr;
  WG_CHECK(cudaMalloc(&dm, kRounds * 32 * 4));
  WG_CHECK(cudaMemset(dm, 0, kRounds * 32 * 4));
  k_mp<<<1, 64>>>(dm);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> m(kRounds * 32);
  WG_CHECK(cudaMemcpy(m.data(), dm, m.size() * 4, cudaMemcpyDeviceToHost));
  {
    Checker ck("async.fence.message_passing");
    for (int r = 0; r < kRounds; ++r) {
      for (uint32_t l = 0; l < 32; ++l) ck.eq(r * 32 + l, m[r * 32 + l], mp_val(r, (l + 3) % 32));
    }
    wg_report(ck);
  }
  WG_CHECK(cudaFree(d));
  WG_CHECK(cudaFree(dm));
  return wg_finish();
}
