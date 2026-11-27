// Tensor (TMA) im2col-mode end-to-end test: cuTensorMapEncodeIm2col (fetched through the
// unversioned cudaGetDriverEntryPoint path) through the shim, im2col loads with offsets, an
// im2col prefetch and an .im2col_no_offs tensor store on an NHWC u16 tensor C=64 W=9 H=14 N=2.
//
// CPU reference of the im2col traversal (ptx.txt 3267-3406, figures 13-16; cuda.h
// cuTensorMapEncodeIm2col): pixel p loads channels [c, c + channelsPerPixel) of the spatial
// position (w + offW, h + offH) of image n; the traversal starts at the tensor coordinates,
// steps W by its element stride, wraps W to the lower corner after W - 1 + upperW and carries
// into H (same rule) and then into N. Out-of-bound pixels / channels are zero-filled on loads
// and skipped on stores. Rows are dense (channelsPerPixel elements) and swizzled on the absolute
// shared-memory address like tile mode.
#include "async_common.cuh"

constexpr int kThreads = 128;
constexpr int kC = 64, kW = 9, kH = 14, kN = 2;

struct Im2colArgs {
  int c, w, h, n;
  uint16_t ow, oh;
  uint32_t bytes;
};

__global__ void k_im2col_load(const __grid_constant__ CUtensorMap tm, Im2colArgs a, uint32_t* out) {
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
    asm volatile(
        "cp.async.bulk.prefetch.tensor.4d.L2.global.im2col [%0, {%1, %2, %3, %4}], {%5, %6};" ::"l"(
            tmap),
        "r"(a.c), "r"(a.w), "r"(a.h), "r"(a.n), "h"(a.ow), "h"(a.oh)
        : "memory");
    mbar_arrive_expect_tx(b, a.bytes);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6], {%7, %8};" ::"r"(dst),
        "l"(tmap), "r"(a.c), "r"(a.w), "r"(a.h), "r"(a.n), "r"(b), "h"(a.ow), "h"(a.oh)
        : "memory");
  }
  mbar_wait(b, 0);
  for (uint32_t i = t; i < a.bytes / 4; i += blockDim.x) out[i] = sw[i];
}

// Store pixel rows: element k of row p is 0x4000 | (p * cpp + k).
__global__ void k_im2col_store(const __grid_constant__ CUtensorMap tm, int c, int w, int h, int n,
                               uint32_t nelem) {
  extern __shared__ unsigned char dyn[];
  unsigned char* s = align1024(dyn);
  const uint32_t t = threadIdx.x;
  const uint64_t tmap = reinterpret_cast<uint64_t>(&tm);
  for (uint32_t i = t; i < nelem; i += blockDim.x) {
    reinterpret_cast<uint16_t*>(s)[i] = (uint16_t)(0x4000u | i);
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (t == 0) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];" ::"l"(tmap),
        "r"(c), "r"(w), "r"(h), "r"(n), "r"(smem_u32(s))
        : "memory");
    bulk_commit();
    bulk_wait0();
  }
}

// ---- host ---------------------------------------------------------------------------------------
static uint16_t xval(int c, int w, int h, int n) {
  return (uint16_t)(((((n * kH + h) * kW + w) * kC + c) * 3 + 7) & 0xFFFF);
}

struct Im2colCase {
  const char* name;
  int lower[2], upper[2];  // index 0 = W, 1 = H
  uint32_t ppc, cpp;
  int c, w, h, n;
  uint16_t ow, oh;
  uint32_t es[4];  // C (ignored), W, H, N
  CUtensorMapSwizzle swz;
};

// Visits the pixels of the traversal; f(p, in, aw, ah, img).
template <class F>
static void traverse(const Im2colCase& k, F f) {
  int64_t pw = k.w, ph = k.h, img = k.n;
  for (uint32_t p = 0; p < k.ppc; ++p) {
    const int64_t aw = pw + k.ow, ah = ph + k.oh;
    const bool in = img >= 0 && img < kN && aw >= 0 && aw < kW && ah >= 0 && ah < kH;
    f(p, in, (int)aw, (int)ah, (int)img);
    pw += k.es[1];
    if (pw <= kW - 1 + k.upper[0]) continue;
    pw = k.lower[0];
    ph += k.es[2];
    if (ph <= kH - 1 + k.upper[1]) continue;
    ph = k.lower[1];
    ++img;
  }
}

static EncodeIm2colFn g_encode = nullptr;

static CUtensorMap encode(void* base, const Im2colCase& k) {
  const cuuint64_t dims[4] = {kC, kW, kH, kN};
  const cuuint64_t strides[3] = {kC * 2, kC * kW * 2, kC * kW * kH * 2};
  CUtensorMap tm;
  WG_CU_CHECK(g_encode(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 4, base, dims, strides, k.lower,
                       k.upper, k.cpp, k.ppc, k.es, CU_TENSOR_MAP_INTERLEAVE_NONE, k.swz,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

int main() {
  // Unversioned (deprecated) runtime entry-point query: cudart passes its own version to the
  // driver's cuGetProcAddress, which the shim answers.
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  WG_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeIm2col", &fn, cudaEnableDefault, &q));
  if (q != cudaDriverEntryPointSuccess || fn == nullptr) {
    std::printf("cuTensorMapEncodeIm2col entry point not found (status %d)\n", (int)q);
    return 2;
  }
  g_encode = reinterpret_cast<EncodeIm2colFn>(fn);

  std::vector<uint16_t> hx((size_t)kC * kW * kH * kN);
  for (int n = 0; n < kN; ++n)
    for (int h = 0; h < kH; ++h)
      for (int w = 0; w < kW; ++w)
        for (int c = 0; c < kC; ++c)
          hx[(((size_t)n * kH + h) * kW + w) * kC + c] = xval(c, w, h, n);
  void* dx = nullptr;
  WG_CHECK(cudaMalloc(&dx, hx.size() * 2));
  WG_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  uint32_t* d_out = nullptr;
  WG_CHECK(cudaMalloc(&d_out, 4096));

  // A: plain 3x3-style box, crosses H rows. B: offsets, shrunk box, 32B swizzle, image 1.
  // C: wraps past the last row of image 0 into image 1 and then into the missing image 2 (OOB).
  // D: element strides 2 on W and H with offsets [inferred: figure 15, traversal steps by the
  // stride and wraps to the lower corner].
  const Im2colCase cases[4] = {
      {"async.tma.im2col.basic",
       {-1, -1},
       {-1, -1},
       64,
       8,
       8,
       7,
       4,
       0,
       0,
       0,
       {1, 1, 1, 1},
       CU_TENSOR_MAP_SWIZZLE_NONE},
      {"async.tma.im2col.offsets_swz32",
       {0, 0},
       {-2, -2},
       40,
       16,
       16,
       3,
       2,
       1,
       2,
       1,
       {1, 1, 1, 1},
       CU_TENSOR_MAP_SWIZZLE_32B},
      {"async.tma.im2col.image_carry_oob",
       {-1, -1},
       {-1, -1},
       160,
       8,
       0,
       5,
       12,
       0,
       0,
       0,
       {1, 1, 1, 1},
       CU_TENSOR_MAP_SWIZZLE_NONE},
      {"async.tma.im2col.element_stride",
       {-1, -1},
       {-1, -1},
       32,
       16,
       16,
       7,
       5,
       0,
       1,
       1,
       {1, 2, 2, 1},
       CU_TENSOR_MAP_SWIZZLE_NONE},
  };
  for (const Im2colCase& k : cases) {
    const CUtensorMap tm = encode(dx, k);
    const uint32_t bytes = k.ppc * k.cpp * 2;
    Im2colArgs a{k.c, k.w, k.h, k.n, k.ow, k.oh, bytes};
    WG_CHECK(cudaMemset(d_out, 0, 4096));
    k_im2col_load<<<1, kThreads, bytes + 1024>>>(tm, a, d_out);
    WG_CHECK(cudaGetLastError());
    WG_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> img(bytes);
    WG_CHECK(cudaMemcpy(img.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    Checker ck(k.name);
    traverse(k, [&](uint32_t p, bool in, int aw, int ah, int n) {
      for (uint32_t e = 0; e < k.cpp; ++e) {
        const int ch = k.c + (int)e;
        const uint16_t want = in && ch < kC ? xval(ch, aw, ah, n) : 0;
        uint16_t got;
        memcpy(&got, img.data() + wg_swizzle((p * k.cpp + e) * 2u, k.swz), 2);
        ck.eq((long)(p * k.cpp + e), got, want);
      }
    });
    wg_report(ck);
  }

  // .im2col_no_offs store: 24 pixels x 16 channels from (c 16, w 2, h 3, n 1); the pixels at
  // w = -1 fall outside the tensor and are not written.
  {
    const Im2colCase k = {"async.tma.im2col.store_no_offs",
                          {-1, -1},
                          {-1, -1},
                          24,
                          16,
                          16,
                          2,
                          3,
                          1,
                          0,
                          0,
                          {1, 1, 1, 1},
                          CU_TENSOR_MAP_SWIZZLE_NONE};
    void* dy = nullptr;
    WG_CHECK(cudaMalloc(&dy, hx.size() * 2));
    WG_CHECK(cudaMemcpy(dy, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    const CUtensorMap tm = encode(dy, k);
    k_im2col_store<<<1, kThreads, k.ppc * k.cpp * 2 + 1024>>>(tm, k.c, k.w, k.h, k.n,
                                                              k.ppc * k.cpp);
    WG_CHECK(cudaGetLastError());
    WG_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> want = hx, got(hx.size());
    traverse(k, [&](uint32_t p, bool in, int aw, int ah, int n) {
      if (!in) return;
      for (uint32_t e = 0; e < k.cpp; ++e) {
        want[(((size_t)n * kH + ah) * kW + aw) * kC + k.c + e] =
            (uint16_t)(0x4000u | (p * k.cpp + e));
      }
    });
    WG_CHECK(cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost));
    Checker ck(k.name);
    for (size_t i = 0; i < want.size(); ++i) ck.eq((long)i, got[i], want[i]);
    wg_report(ck);
    WG_CHECK(cudaFree(dy));
  }

  WG_CHECK(cudaFree(dx));
  WG_CHECK(cudaFree(d_out));
  return wg_finish();
}
