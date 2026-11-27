// tensormap.replace / tensormap.query / tensormap.cp_fenceproxy / fence.proxy.tensormap::generic
// end-to-end test (ptx.txt 16807-17150, 21938-21990, 19756).
//
// Map X: u32 64 x 32 (row stride 256 bytes), box 8 x 4, no swizzle, built by the shim encoder.
// Tensor Y: u32 48 x 40 (row stride 192), value 0x10000 + 256 y + x.
//
// Kernel 1 (one warp): copy X into shared memory, rewrite it into a map of Y with a box 8 x 8 and
// 32B swizzle through every .field (shared::cta and generic addressing, alloc classes), publish
// it to global memory with tensormap.cp_fenceproxy, acquire it and load box (44, 36) through it.
// Kernel 2: rewrite a global-memory copy of X in place (.global and generic addressing), release /
// acquire through fence.proxy.tensormap::generic and load box (40, 38).
#include "async_common.cuh"

constexpr uint32_t kYw = 48, kYh = 40;

__device__ __forceinline__ uint32_t tm_query_shared(uint32_t sa) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "tensormap.query.alloc_class_large.shared::cta.b1024 p, [%1];\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(r)
      : "r"(sa)
      : "memory");
  return r;
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* tmap, int x, int y,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(tmap), "r"(x), "r"(y), "r"(bar)
      : "memory");
}

// Loads 8 x 8 u32 (256 bytes) through `tmap` into a 1024-aligned buffer and copies it out.
__device__ void load_box(const void* tmap, int x, int y, uint32_t* out) {
  __shared__ alignas(1024) uint32_t box[64];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  for (uint32_t i = t; i < 64; i += blockDim.x) box[i] = 0xDEADBEEFu;
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b, 256);
    tma_load_2d(smem_u32(box), tmap, x, y, b);
  }
  mbar_wait(b, 0);
  for (uint32_t i = t; i < 64; i += blockDim.x) out[i] = box[i];
}

__global__ void k_replace_shared(const __grid_constant__ CUtensorMap tm, void* ybase,
                                 CUtensorMap* gtm, uint32_t* out, uint32_t* q) {
  __shared__ alignas(128) CUtensorMap stm;
  const uint32_t lane = threadIdx.x;
  reinterpret_cast<uint32_t*>(&stm)[lane] = reinterpret_cast<const uint32_t*>(&tm)[lane];
  __syncwarp();
  const uint32_t sa = smem_u32(&stm);
  if (lane == 0) {
    uint32_t r;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "tensormap.query.alloc_class_large.b1024 p, [%1];\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(r)
        : "l"(&tm)
        : "memory");
    q[0] = r;
    q[1] = tm_query_shared(sa);
    asm volatile(
        "tensormap.replace.tile.global_address.alloc_large.shared::cta.b1024.b64 [%0], %1;" ::"r"(
            sa),
        "l"(ybase)
        : "memory");
    q[2] = tm_query_shared(sa);
    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"r"(sa),
                 "r"(kYw)
                 : "memory");
    asm volatile("tensormap.replace.tile.global_dim.b1024.b32 [%0], 1, %1;" ::"l"(&stm), "r"(kYh)
                 : "memory");
    asm volatile(
        "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" ::"r"(sa),
        "l"((uint64_t)kYw * 4)
        : "memory");
    asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(sa),
                 "r"(8u)
                 : "memory");
    asm volatile(
        "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(sa),
        "r"(1u)
        : "memory");
    asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], %1;" ::"r"(sa), "r"(1u)
                 : "memory");
    asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 2;" ::"r"(sa)
                 : "memory");
    asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0;" ::"r"(sa)
                 : "memory");
    asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 1;" ::"r"(sa)
                 : "memory");
    asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0;" ::"r"(sa)
                 : "memory");
    asm volatile("tensormap.replace.tile.alloc_small.shared::cta.b1024 [%0];" ::"r"(sa) : "memory");
    q[3] = tm_query_shared(sa);
    asm volatile("tensormap.replace.tile.alloc_large.shared::cta.b1024 [%0];" ::"r"(sa) : "memory");
    q[4] = tm_query_shared(sa);
    asm volatile(
        "tensormap.replace.tile.global_address.alloc_same.shared::cta.b1024.b64 [%0], %1;" ::"r"(
            sa),
        "l"(ybase)
        : "memory");
    q[5] = tm_query_shared(sa);
  }
  __syncwarp();
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"
      " [%0], [%1], 128;" ::"l"(gtm),
      "r"(sa)
      : "memory");
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(gtm) : "memory");
  load_box(gtm, 44, 36, out);
}

__global__ void k_replace_global(CUtensorMap* gtm, void* ybase, uint32_t* out, uint32_t* q) {
  if (threadIdx.x == 0) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;" ::"l"(gtm),
                 "l"(ybase)
                 : "memory");
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 0, %1;" ::"l"(gtm),
                 "r"(kYw)
                 : "memory");
    asm volatile("tensormap.replace.tile.global_dim.b1024.b32 [%0], 1, %1;" ::"l"(gtm), "r"(kYh)
                 : "memory");
    asm volatile("tensormap.replace.tile.global_stride.b1024.b64 [%0], 0, %1;" ::"l"(gtm),
                 "l"((uint64_t)kYw * 4)
                 : "memory");
    asm volatile("tensormap.replace.tile.box_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(gtm), "r"(8u)
                 : "memory");
    uint32_t r;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "tensormap.query.alloc_class_large.global.b1024 p, [%1];\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(r)
        : "l"(gtm)
        : "memory");
    q[0] = r;
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
  }
  __syncthreads();
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(gtm) : "memory");
  load_box(gtm, 40, 38, out);
}

static uint32_t yval(uint32_t x, uint32_t y) { return 0x10000u + y * 256u + x; }

static void check_box(const char* name, const std::vector<uint32_t>& got, int x0, int y0,
                      CUtensorMapSwizzle swz) {
  Checker ck(name);
  for (int k1 = 0; k1 < 8; ++k1) {
    for (int k0 = 0; k0 < 8; ++k0) {
      const int x = x0 + k0, y = y0 + k1;
      const uint32_t want = (x < (int)kYw && y < (int)kYh) ? yval(x, y) : 0u;
      const uint32_t off = wg_swizzle((uint32_t)(k1 * 8 + k0) * 4u, swz);
      ck.eq(k1 * 8 + k0, got[off / 4], want);
    }
  }
  wg_report(ck);
}

int main() {
  void *dx = nullptr, *dy = nullptr;
  WG_CHECK(cudaMalloc(&dx, 64 * 32 * 4));
  WG_CHECK(cudaMemset(dx, 0x5A, 64 * 32 * 4));
  std::vector<uint32_t> hy(kYw * kYh);
  for (uint32_t y = 0; y < kYh; ++y)
    for (uint32_t x = 0; x < kYw; ++x) hy[y * kYw + x] = yval(x, y);
  WG_CHECK(cudaMalloc(&dy, hy.size() * 4));
  WG_CHECK(cudaMemcpy(dy, hy.data(), hy.size() * 4, cudaMemcpyHostToDevice));

  CUtensorMap xm;
  const cuuint64_t dims[2] = {64, 32};
  const cuuint64_t strides[1] = {256};
  const cuuint32_t box[2] = {8, 4};
  const cuuint32_t es[2] = {1, 1};
  wg_make_tiled(&xm, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, dx, dims, strides, box, es,
                CU_TENSOR_MAP_SWIZZLE_NONE);

  CUtensorMap* gtm = nullptr;
  uint32_t *d_out = nullptr, *d_q = nullptr;
  WG_CHECK(cudaMalloc(&gtm, 2 * sizeof(CUtensorMap)));
  WG_CHECK(cudaMalloc(&d_out, 64 * 4));
  WG_CHECK(cudaMalloc(&d_q, 8 * 4));
  std::vector<uint32_t> out(64), q(8);

  // Kernel 1.
  WG_CHECK(cudaMemset(gtm, 0, 2 * sizeof(CUtensorMap)));
  WG_CHECK(cudaMemset(d_q, 0xFF, 8 * 4));
  k_replace_shared<<<1, 32>>>(xm, dy, gtm, d_out, d_q);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  WG_CHECK(cudaMemcpy(out.data(), d_out, 64 * 4, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(q.data(), d_q, 8 * 4, cudaMemcpyDeviceToHost));
  check_box("async.tensormap.replace_shared_cp_fenceproxy", out, 44, 36, CU_TENSOR_MAP_SWIZZLE_32B);
  {
    // param map small, copy small, after .global_address.alloc_large large, after
    // .alloc_small small, after .alloc_large large, after .global_address.alloc_same unchanged.
    const uint32_t want[6] = {0, 0, 1, 0, 1, 1};
    Checker ck("async.tensormap.alloc_class_query");
    for (int i = 0; i < 6; ++i) ck.eq(i, q[i], want[i]);
    wg_report(ck);
  }

  // Kernel 2: in-place rewrite of a global copy of X.
  WG_CHECK(cudaMemcpy(gtm + 1, &xm, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_q, 0xFF, 8 * 4));
  k_replace_global<<<1, 64>>>(gtm + 1, dy, d_out, d_q);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  WG_CHECK(cudaMemcpy(out.data(), d_out, 64 * 4, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(q.data(), d_q, 8 * 4, cudaMemcpyDeviceToHost));
  check_box("async.tensormap.replace_global", out, 40, 38, CU_TENSOR_MAP_SWIZZLE_NONE);
  {
    // .global_address without a class sets the allocation size to small (ptx.txt 17066-17068).
    Checker ck("async.tensormap.global_address_resets_alloc_class");
    ck.eq(0, q[0], 0u);
    wg_report(ck);
  }

  WG_CHECK(cudaFree(dx));
  WG_CHECK(cudaFree(dy));
  WG_CHECK(cudaFree(gtm));
  WG_CHECK(cudaFree(d_out));
  WG_CHECK(cudaFree(d_q));
  return wg_finish();
}
