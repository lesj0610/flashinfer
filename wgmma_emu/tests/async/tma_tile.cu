// Tensor (TMA) tile-mode end-to-end test: cuTensorMapEncodeTiled through the shim, then
// cp.async.bulk.tensor loads (2D / 3D, every swizzle mode, OOB zero fill, OOB NaN fill, element
// strides, .shared::cta destination, multicast to the own CTA, L2 cache hint, prefetches),
// a tensor store and tensor reductions (add, max with cache hint).
//
// Loads: the device copies the raw shared-memory image (1024-byte aligned buffer) back, and the
// host compares it with a CPU reference that places element i of the dense box at the
// swizzled byte offset (ptx.txt 3210-3248 tile mode and bounding box, 3460-3540 OOB fill,
// 3552-4038 swizzling; cuda.h 23305-23310 element strides, dimension 0 ignored without
// interleave). Store / reduce: the host checks the whole destination tensor.
#include <cmath>

#include "async_common.cuh"

constexpr int kThreads = 128;

struct LoadArgs {
  int variant;  // 0: 2d .shared::cluster (+ prefetches), 1: 2d .shared::cta, 2: 2d multicast,
                // 3: 3d, 4: 2d .L2::cache_hint
  int c0, c1, c2;
  uint32_t bytes;
};

__global__ void k_tma_load(const __grid_constant__ CUtensorMap tm, LoadArgs a, uint32_t* out) {
  extern __shared__ unsigned char dyn[];
  __shared__ alignas(8) uint64_t bar;
  unsigned char* s = align1024(dyn);
  uint32_t* sw = reinterpret_cast<uint32_t*>(s);
  const uint32_t t = threadIdx.x;
  const uint32_t dst = smem_u32(s);
  const uint32_t b = smem_u32(&bar);
  const uint64_t tmap = reinterpret_cast<uint64_t>(&tm);
  for (uint32_t i = t; i < a.bytes / 4; i += blockDim.x) sw[i] = 0xDEADBEEFu;
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b, a.bytes);
    switch (a.variant) {
      case 0:
        asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(tmap),
                     "r"(a.c0), "r"(a.c1)
                     : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
            "l"(tmap), "r"(a.c0), "r"(a.c1), "r"(b)
            : "memory");
        break;
      case 1:
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
            "l"(tmap), "r"(a.c0), "r"(a.c1), "r"(b)
            : "memory");
        break;
      case 2: {
        const uint16_t mask = 1;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
            "l"(tmap), "r"(a.c0), "r"(a.c1), "r"(b), "h"(mask)
            : "memory");
        break;
      }
      case 3:
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
            "l"(tmap), "r"(a.c0), "r"(a.c1), "r"(a.c2), "r"(b)
            : "memory");
        break;
      default: {
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
            "l"(tmap), "r"(a.c0), "r"(a.c1), "r"(b), "l"(pol)
            : "memory");
        break;
      }
    }
  }
  mbar_wait(b, 0);
  for (uint32_t i = t; i < a.bytes / 4; i += blockDim.x) out[i] = sw[i];
}

// kind 0: tensor store of u16 0x8000 | i (swizzled); 1: tensor reduce .add of u32 i + 1;
// 2: tensor reduce .max of u32 hash(i + 900) with an L2 cache hint.
__global__ void k_tma_s2g(const __grid_constant__ CUtensorMap tm, int kind, int c0, int c1,
                          uint32_t nelem, CUtensorMapSwizzle swz) {
  extern __shared__ unsigned char dyn[];
  unsigned char* s = align1024(dyn);
  const uint32_t t = threadIdx.x;
  const uint64_t tmap = reinterpret_cast<uint64_t>(&tm);
  for (uint32_t i = t; i < nelem; i += blockDim.x) {
    if (kind == 0) {
      *reinterpret_cast<uint16_t*>(s + wg_swizzle(2u * i, swz)) = (uint16_t)(0x8000u | i);
    } else {
      *reinterpret_cast<uint32_t*>(s + wg_swizzle(4u * i, swz)) =
          kind == 1 ? i + 1u : wg_hash32(i + 900u);
    }
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (t == 0) {
    const uint32_t src = smem_u32(s);
    if (kind == 0) {
      asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
              tmap),
          "r"(c0), "r"(c1), "r"(src)
          : "memory");
    } else if (kind == 1) {
      asm volatile(
          "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
          " [%0, {%1, %2}], [%3];" ::"l"(tmap),
          "r"(c0), "r"(c1), "r"(src)
          : "memory");
    } else {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      asm volatile(
          "cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
          " [%0, {%1, %2}], [%3], %4;" ::"l"(tmap),
          "r"(c0), "r"(c1), "r"(src), "l"(pol)
          : "memory");
    }
    bulk_commit();
    bulk_wait_read0();
    bulk_wait0();
  }
}

// ---- host ---------------------------------------------------------------------------------------
struct Tensor {
  CUtensorMapDataType dt;
  int rank;
  int esize;
  cuuint64_t dim[3];
  std::vector<uint8_t> host;  // dense, dimension 0 fastest
  void* dev = nullptr;
  uint64_t elem(uint64_t x, uint64_t y, uint64_t z) const {
    const uint64_t i = (z * dim[1] + y) * dim[0] + x;
    uint64_t v = 0;
    memcpy(&v, host.data() + i * esize, esize);
    return v;
  }
};

static void upload(Tensor& t) {
  WG_CHECK(cudaMalloc(&t.dev, t.host.size()));
  WG_CHECK(cudaMemcpy(t.dev, t.host.data(), t.host.size(), cudaMemcpyHostToDevice));
}

static CUtensorMap encode(const Tensor& t, const cuuint32_t* box, const cuuint32_t* es,
                          CUtensorMapSwizzle swz,
                          CUtensorMapFloatOOBfill fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  cuuint64_t stride[2] = {t.dim[0] * t.esize, t.dim[0] * t.dim[1] * t.esize};
  CUtensorMap tm;
  wg_make_tiled(&tm, t.dt, t.rank, t.dev, t.dim, stride, box, es, swz, fill);
  return tm;
}

static uint32_t* g_out = nullptr;

static void run_load(const char* name, const Tensor& t, const CUtensorMap& tm, int variant,
                     const int* c, const cuuint32_t* box, const cuuint32_t* es,
                     CUtensorMapSwizzle swz, bool nan_fill) {
  uint32_t n[3] = {1, 1, 1}, step[3] = {1, 1, 1};
  for (int d = 0; d < t.rank; ++d) {
    step[d] = d == 0 ? 1u : es[d];
    n[d] = (box[d] + step[d] - 1) / step[d];
  }
  const uint32_t bytes = n[0] * n[1] * n[2] * (uint32_t)t.esize;
  LoadArgs a{variant, c[0], c[1], t.rank > 2 ? c[2] : 0, bytes};
  WG_CHECK(cudaMemset(g_out, 0, 4096));
  k_tma_load<<<1, kThreads, bytes + 1024>>>(tm, a, g_out);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> img(bytes);
  WG_CHECK(cudaMemcpy(img.data(), g_out, bytes, cudaMemcpyDeviceToHost));

  Checker ck(name);
  long idx = 0;
  for (uint32_t k2 = 0; k2 < n[2]; ++k2) {
    for (uint32_t k1 = 0; k1 < n[1]; ++k1) {
      for (uint32_t k0 = 0; k0 < n[0]; ++k0, ++idx) {
        const int64_t x = (int64_t)c[0] + k0;
        const int64_t y = (int64_t)c[1] + (int64_t)k1 * step[1];
        const int64_t z = t.rank > 2 ? (int64_t)c[2] + (int64_t)k2 * step[2] : 0;
        const bool in = x >= 0 && x < (int64_t)t.dim[0] && y >= 0 && y < (int64_t)t.dim[1] &&
                        z >= 0 && (t.rank < 3 || z < (int64_t)t.dim[2]);
        const uint32_t off = wg_swizzle((uint32_t)idx * (uint32_t)t.esize, swz);
        uint64_t got = 0;
        memcpy(&got, img.data() + off, t.esize);
        if (!in && nan_fill) {
          float f;
          memcpy(&f, &got, 4);
          ck.ok(idx, std::isnan(f));
        } else {
          ck.eq(idx, got, in ? t.elem((uint64_t)x, (uint64_t)y, (uint64_t)z) : 0u);
        }
      }
    }
  }
  wg_report(ck);
}

int main() {
  WG_CHECK(cudaMalloc(&g_out, 4096));

  // U16 2D tensor 104 x 70.
  Tensor A{CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, 2, {104, 70, 1}, {}};
  A.host.resize(104 * 70 * 2);
  for (int y = 0; y < 70; ++y) {
    for (int x = 0; x < 104; ++x) {
      const uint16_t v = (uint16_t)((x * 7 + y * 131 + 1) & 0xFFFF);
      memcpy(&A.host[(y * 104 + x) * 2], &v, 2);
    }
  }
  upload(A);

  // U32 3D tensor 12 x 10 x 6.
  Tensor C{CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, 4, {12, 10, 6}, {}};
  C.host.resize(12 * 10 * 6 * 4);
  for (int z = 0; z < 6; ++z) {
    for (int y = 0; y < 10; ++y) {
      for (int x = 0; x < 12; ++x) {
        const uint32_t v = (uint32_t)(x + 100 * y + 10000 * z + 1);
        memcpy(&C.host[((z * 10 + y) * 12 + x) * 4], &v, 4);
      }
    }
  }
  upload(C);

  // F32 2D tensor 20 x 9.
  Tensor F{CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, 4, {20, 9, 1}, {}};
  F.host.resize(20 * 9 * 4);
  for (int y = 0; y < 9; ++y) {
    for (int x = 0; x < 20; ++x) {
      const float v = (float)x + 0.25f * (float)y;
      memcpy(&F.host[(y * 20 + x) * 4], &v, 4);
    }
  }
  upload(F);

  const cuuint32_t es1[3] = {1, 1, 1};
  {
    const cuuint32_t box[2] = {40, 16};
    const int c[2] = {8, 4};
    run_load("async.tma.tile2d.none.shared_cta", A, encode(A, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE),
             1, c, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE, false);
  }
  {
    const cuuint32_t box[2] = {16, 16};
    const int c[2] = {96, 60};
    run_load("async.tma.tile2d.swz32.oob_high.prefetch", A,
             encode(A, box, es1, CU_TENSOR_MAP_SWIZZLE_32B), 0, c, box, es1,
             CU_TENSOR_MAP_SWIZZLE_32B, false);
  }
  {
    const cuuint32_t box[2] = {32, 8};
    const int c[2] = {-5, -2};
    run_load("async.tma.tile2d.swz64.oob_low.cache_hint", A,
             encode(A, box, es1, CU_TENSOR_MAP_SWIZZLE_64B), 4, c, box, es1,
             CU_TENSOR_MAP_SWIZZLE_64B, false);
  }
  {
    const cuuint32_t box[2] = {64, 16};
    const int c[2] = {40, 20};
    run_load("async.tma.tile2d.swz128", A, encode(A, box, es1, CU_TENSOR_MAP_SWIZZLE_128B), 0, c,
             box, es1, CU_TENSOR_MAP_SWIZZLE_128B, false);
  }
  {
    // elementStrides[0] = 2 is ignored without interleave (cuda.h 23305-23310).
    const cuuint32_t box[2] = {40, 16};
    const cuuint32_t es[2] = {2, 2};
    const int c[2] = {0, 3};
    run_load("async.tma.tile2d.element_stride", A, encode(A, box, es, CU_TENSOR_MAP_SWIZZLE_NONE),
             0, c, box, es, CU_TENSOR_MAP_SWIZZLE_NONE, false);
  }
  {
    const cuuint32_t box[2] = {64, 8};
    const int c[2] = {0, 0};
    run_load("async.tma.tile2d.multicast_self", A, encode(A, box, es1, CU_TENSOR_MAP_SWIZZLE_128B),
             2, c, box, es1, CU_TENSOR_MAP_SWIZZLE_128B, false);
  }
  {
    const cuuint32_t box[3] = {8, 4, 3};
    const int c[3] = {6, 7, 4};
    run_load("async.tma.tile3d.u32.oob", C, encode(C, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE), 3, c,
             box, es1, CU_TENSOR_MAP_SWIZZLE_NONE, false);
  }
  {
    const cuuint32_t box[2] = {8, 4};
    const int c[2] = {16, 7};
    run_load("async.tma.tile2d.f32.nan_fill", F,
             encode(F, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA),
             0, c, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE, true);
  }

  // Tensor store: box 64 x 8, 128B swizzle, partly outside the 104 x 70 tensor.
  {
    Tensor S = A;
    S.dev = nullptr;
    upload(S);
    const cuuint32_t box[2] = {64, 8};
    const CUtensorMap tm = encode(S, box, es1, CU_TENSOR_MAP_SWIZZLE_128B);
    k_tma_s2g<<<1, kThreads, 2048>>>(tm, 0, 50, 66, 64 * 8, CU_TENSOR_MAP_SWIZZLE_128B);
    WG_CHECK(cudaGetLastError());
    WG_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(104 * 70), want(104 * 70);
    WG_CHECK(cudaMemcpy(got.data(), S.dev, got.size() * 2, cudaMemcpyDeviceToHost));
    memcpy(want.data(), A.host.data(), want.size() * 2);
    for (int k1 = 0; k1 < 8; ++k1) {
      for (int k0 = 0; k0 < 64; ++k0) {
        const int x = 50 + k0, y = 66 + k1;
        if (x < 104 && y < 70) want[y * 104 + x] = (uint16_t)(0x8000 | (k1 * 64 + k0));
      }
    }
    Checker ck("async.tma.store2d.swz128.oob");
    for (size_t i = 0; i < want.size(); ++i) ck.eq((long)i, got[i], want[i]);
    wg_report(ck);
    WG_CHECK(cudaFree(S.dev));
  }

  // Tensor reductions on a U32 24 x 10 tensor: .add at (20, 8), then .max at (0, 0).
  {
    Tensor R{CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, 4, {24, 10, 1}, {}};
    R.host.resize(24 * 10 * 4);
    std::vector<uint32_t> want(24 * 10);
    for (int i = 0; i < 24 * 10; ++i) {
      want[i] = wg_hash32((uint32_t)i);
      memcpy(&R.host[i * 4], &want[i], 4);
    }
    upload(R);
    const cuuint32_t box[2] = {8, 4};
    const CUtensorMap tm = encode(R, box, es1, CU_TENSOR_MAP_SWIZZLE_NONE);
    k_tma_s2g<<<1, kThreads, 2048>>>(tm, 1, 20, 8, 32, CU_TENSOR_MAP_SWIZZLE_NONE);
    WG_CHECK(cudaGetLastError());
    k_tma_s2g<<<1, kThreads, 2048>>>(tm, 2, 0, 0, 32, CU_TENSOR_MAP_SWIZZLE_NONE);
    WG_CHECK(cudaGetLastError());
    WG_CHECK(cudaDeviceSynchronize());
    for (int k1 = 0; k1 < 4; ++k1) {
      for (int k0 = 0; k0 < 8; ++k0) {
        const uint32_t i = (uint32_t)(k1 * 8 + k0);
        const int x = 20 + k0, y = 8 + k1;
        if (x < 24 && y < 10) want[y * 24 + x] += i + 1u;
        uint32_t& m = want[k1 * 24 + k0];
        const uint32_t v = wg_hash32(i + 900u);
        m = m > v ? m : v;
      }
    }
    std::vector<uint32_t> got(24 * 10);
    WG_CHECK(cudaMemcpy(got.data(), R.dev, got.size() * 4, cudaMemcpyDeviceToHost));
    Checker ck("async.tma.reduce2d.add_max");
    for (size_t i = 0; i < want.size(); ++i) ck.eq((long)i, got[i], want[i]);
    wg_report(ck);
    WG_CHECK(cudaFree(R.dev));
  }

  WG_CHECK(cudaFree(A.dev));
  WG_CHECK(cudaFree(C.dev));
  WG_CHECK(cudaFree(F.dev));
  WG_CHECK(cudaFree(g_out));
  return wg_finish();
}
