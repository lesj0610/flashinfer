// E2 (WGEMU_COUNT=1) probe with hand-countable work; the manifest lists the expected
// WGEMU-COUNT lines. Also a positive test of the exact and delay modes.
//
// k_count_wgmma, grid 2 x 256 threads (two warpgroups per CTA), per thread:
//   one ld.global.nc.v4.u32 (16 bytes) and one ld.global.u32 (4 bytes)   -> ldg_bytes 512 * 20
//   4 x wgmma m64n8k16.f32.bf16.bf16 and 1 x wgmma m64n16k16.f32.bf16.bf16, counted once per
//   warpgroup                                     -> 4 warpgroups: 16 and 4
// A and B are bf16 1.0 everywhere, so every accumulator element is 16 per wgmma whatever the
// descriptor layout: 64 after the four n8 steps (first one with scale-d 0), 16 for n16.
// k_count_tma, grid 1 x 128 threads: two 2D tile loads of a 32 x 8 box of u32 (2 x 1024 bytes,
// tma_bytes 2048) completing on one mbarrier that all 128 threads wait for (mbar_waits 128).
#include "async_common.cuh"

constexpr int kWgThreads = 256;
constexpr int kWgBlocks = 2;
constexpr int kWgTotal = kWgThreads * kWgBlocks;
constexpr int kOutPerThread = 4 + 8 + 1;
constexpr int kOnes = 4096;  // bf16 elements of shared memory set to 1.0 (8 KB)

__device__ __forceinline__ uint64_t desc_nosw(uint32_t addr) {
  // start address >> 4, leading byte offset 128, stride byte offset 256, no swizzle.
  return (uint64_t)((addr >> 4) & 0x3FFFu) | ((uint64_t)(128u >> 4) << 16) |
         ((uint64_t)(256u >> 4) << 32);
}

__global__ void k_count_wgmma(const uint32_t* g, uint32_t* out) {
  __shared__ alignas(1024) uint16_t ones[kOnes];
  const uint32_t t = threadIdx.x;
  const uint32_t gt = blockIdx.x * kWgThreads + t;
  for (uint32_t i = t; i < kOnes; i += kWgThreads) ones[i] = 0x3F80u;  // bf16 1.0
  fence_proxy_async_smem();
  __syncthreads();

  uint32_t v0, v1, v2, v3, u;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
               : "l"(g + 4u * gt));
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(u) : "l"(g + 4u * kWgTotal + gt));

  const uint64_t adesc = desc_nosw(smem_u32(ones));
  const uint64_t bdesc = desc_nosw(smem_u32(ones + kOnes / 2));
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < 4; ++k) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, "
        "0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(adesc), "l"(bdesc), "r"(k));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  float e[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
      "%8, %9, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3]), "+f"(e[4]), "+f"(e[5]), "+f"(e[6]),
        "+f"(e[7])
      : "l"(adesc), "l"(bdesc), "r"(0));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

  uint32_t* o = out + kOutPerThread * gt;
  for (int i = 0; i < 4; ++i) o[i] = __float_as_uint(d[i]);
  for (int i = 0; i < 8; ++i) o[4 + i] = __float_as_uint(e[i]);
  o[12] = v0 + v1 + v2 + v3 + u;
}

constexpr int kTmaThreads = 128;
constexpr int kCols = 64, kRows = 16, kBoxX = 32, kBoxY = 8;
constexpr int kBoxWords = kBoxX * kBoxY;

__global__ void k_count_tma(const __grid_constant__ CUtensorMap tm, uint32_t* out) {
  __shared__ alignas(1024) uint32_t s[2 * kBoxWords];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  const uint64_t tmap = reinterpret_cast<uint64_t>(&tm);
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b, 2 * kBoxWords * 4);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(s)),
        "l"(tmap), "r"(0), "r"(0), "r"(b)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(s + kBoxWords)),
        "l"(tmap), "r"(kBoxX), "r"(kBoxY), "r"(b)
        : "memory");
  }
  mbar_wait(b, 0);
  for (uint32_t i = t; i < 2 * kBoxWords; i += kTmaThreads) out[i] = s[i];
}

int main() {
  // ---- wgmma + ld.global ----
  std::vector<uint32_t> hg(5 * kWgTotal);
  for (size_t i = 0; i < hg.size(); ++i) hg[i] = wg_hash32((uint32_t)i + 3u);
  uint32_t *d_g, *d_out;
  WG_CHECK(cudaMalloc(&d_g, 4 * hg.size()));
  WG_CHECK(cudaMalloc(&d_out, 4 * kOutPerThread * kWgTotal));
  WG_CHECK(cudaMemcpy(d_g, hg.data(), 4 * hg.size(), cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_out, 0, 4 * kOutPerThread * kWgTotal));
  k_count_wgmma<<<kWgBlocks, kWgThreads>>>(d_g, d_out);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(kOutPerThread * kWgTotal);
  WG_CHECK(cudaMemcpy(o.data(), d_out, 4 * o.size(), cudaMemcpyDeviceToHost));
  Checker cw("async.count_probe.wgmma_ones");
  Checker cl("async.count_probe.ld_global");
  for (int gt = 0; gt < kWgTotal; ++gt) {
    const uint32_t* r = o.data() + kOutPerThread * gt;
    for (int i = 0; i < 4; ++i) cw.eq(12 * gt + i, r[i], 0x42800000u);          // 64.0f
    for (int i = 0; i < 8; ++i) cw.eq(12 * gt + 4 + i, r[4 + i], 0x41800000u);  // 16.0f
    const uint32_t want =
        hg[4 * gt] + hg[4 * gt + 1] + hg[4 * gt + 2] + hg[4 * gt + 3] + hg[4 * kWgTotal + gt];
    cl.eq(gt, r[12], want);
  }
  wg_report(cw);
  wg_report(cl);
  WG_CHECK(cudaFree(d_g));
  WG_CHECK(cudaFree(d_out));

  // ---- TMA tile loads ----
  std::vector<uint32_t> ht(kCols * kRows);
  for (int i = 0; i < kCols * kRows; ++i) ht[i] = wg_hash32((uint32_t)i + 500u);
  uint32_t *d_t, *d_s;
  WG_CHECK(cudaMalloc(&d_t, 4 * ht.size()));
  WG_CHECK(cudaMalloc(&d_s, 4 * 2 * kBoxWords));
  WG_CHECK(cudaMemcpy(d_t, ht.data(), 4 * ht.size(), cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_s, 0, 4 * 2 * kBoxWords));
  CUtensorMap tm;
  const cuuint64_t dims[2] = {kCols, kRows};
  const cuuint64_t strides[1] = {kCols * 4};
  const cuuint32_t box[2] = {kBoxX, kBoxY};
  const cuuint32_t est[2] = {1, 1};
  wg_make_tiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, d_t, dims, strides, box, est,
                CU_TENSOR_MAP_SWIZZLE_NONE);
  k_count_tma<<<1, kTmaThreads>>>(tm, d_s);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> so(2 * kBoxWords);
  WG_CHECK(cudaMemcpy(so.data(), d_s, 4 * so.size(), cudaMemcpyDeviceToHost));
  Checker ct("async.count_probe.tma_tile");
  for (int y = 0; y < kBoxY; ++y) {
    for (int x = 0; x < kBoxX; ++x) {
      ct.eq(y * kBoxX + x, so[y * kBoxX + x], ht[y * kCols + x]);
      ct.eq(kBoxWords + y * kBoxX + x, so[kBoxWords + y * kBoxX + x],
            ht[(kBoxY + y) * kCols + kBoxX + x]);
    }
  }
  wg_report(ct);
  WG_CHECK(cudaFree(d_t));
  WG_CHECK(cudaFree(d_s));
  return wg_finish();
}
