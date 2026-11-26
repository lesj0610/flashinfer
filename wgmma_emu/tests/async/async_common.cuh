// Shared helpers of the tests/async end-to-end tests (owner A). Every test is built by
// tools/nvcc_emu from sm_90a sources and prints the PLAN section 11 lines:
//   WGEMU-TEST name=<id> cases=<n> mismatches=<m>   ...   WGEMU-TEST-END
// A CUDA error exits with status 2 before WGEMU-TEST-END, so the harness reports a failure.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define WG_CHECK(x)                                                                               \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      std::printf("CUDA error %d (%s) at %s:%d: %s\n", (int)e_, cudaGetErrorString(e_), __FILE__, \
                  __LINE__, #x);                                                                  \
      std::exit(2);                                                                               \
    }                                                                                             \
  } while (0)

#define WG_CU_CHECK(x)                                                                \
  do {                                                                                \
    CUresult r_ = (x);                                                                \
    if (r_ != CUDA_SUCCESS) {                                                         \
      std::printf("driver error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x); \
      std::exit(2);                                                                   \
    }                                                                                 \
  } while (0)

// ---- host-side result bookkeeping ----------------------------------------------------------
struct Checker {
  const char* name;
  long cases = 0;
  long mism = 0;
  explicit Checker(const char* n) : name(n) {}
  void eq(long idx, unsigned long long got, unsigned long long want) {
    ++cases;
    if (got != want) {
      if (mism < 8) {
        std::printf("WGEMU-MISMATCH name=%s idx=%ld got=0x%llx want=0x%llx\n", name, idx, got,
                    want);
      }
      ++mism;
    }
  }
  void ok(long idx, bool cond) { eq(idx, cond ? 1u : 0u, 1u); }
};

inline int& wg_failed_items() {
  static int n = 0;
  return n;
}

inline void wg_report(const Checker& c) {
  std::printf("WGEMU-TEST name=%s cases=%ld mismatches=%ld\n", c.name, c.cases, c.mism);
  if (c.mism != 0 || c.cases == 0) ++wg_failed_items();
}

inline int wg_finish() {
  std::printf("WGEMU-TEST-END\n");
  std::fflush(stdout);
  return wg_failed_items() ? 1 : 0;
}

// ---- tensor-map encoders through the runtime entry-point path --------------------------------
// With a static cudart this goes cudaGetDriverEntryPointByVersion -> driver cuGetProcAddress_v2
// (found by cudart through dlsym on libcuda, which the shim hooks) -> shim encoder. Without the
// shim the SM80 driver encoder returns CUDA_ERROR_NOT_SUPPORTED and the test exits with 2.
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                  const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
typedef CUresult (*EncodeIm2colFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const int*, const int*,
                                   cuuint32_t, cuuint32_t, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion,
                                   CUtensorMapFloatOOBfill);

inline void* wg_driver_entry(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  WG_CHECK(cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &q));
  if (q != cudaDriverEntryPointSuccess || fn == nullptr) {
    std::printf("driver entry point %s not found (status %d)\n", name, (int)q);
    std::exit(2);
  }
  return fn;
}

inline EncodeTiledFn wg_encode_tiled() {
  return reinterpret_cast<EncodeTiledFn>(wg_driver_entry("cuTensorMapEncodeTiled"));
}

// Row-major tiled map with dimension 0 innermost; strides in bytes for dimensions 1..rank-1.
inline void wg_make_tiled(CUtensorMap* tm, CUtensorMapDataType dt, int rank, void* base,
                          const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                          const cuuint32_t* estride, CUtensorMapSwizzle swz,
                          CUtensorMapFloatOOBfill fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  WG_CU_CHECK(wg_encode_tiled()(tm, dt, (cuuint32_t)rank, base, dims, strides, box, estride,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill));
}

// Swizzle<B,4,3> of PTX tensor copies on an absolute shared-memory byte offset (ptx.txt
// 3552-4038): the 16-byte chunk bits [4, 4+B) are XORed with bits [7, 7+B), B = 1/2/3 for the
// 32/64/128-byte modes. Tests align their buffers to 1024 bytes, so offsets equal addresses.
__host__ __device__ inline uint32_t wg_swizzle_bits(CUtensorMapSwizzle swz) {
  return swz == CU_TENSOR_MAP_SWIZZLE_32B    ? 1u
         : swz == CU_TENSOR_MAP_SWIZZLE_64B  ? 2u
         : swz == CU_TENSOR_MAP_SWIZZLE_128B ? 3u
                                             : 0u;
}

__host__ __device__ inline uint32_t wg_swizzle(uint32_t off, CUtensorMapSwizzle swz) {
  const uint32_t b = wg_swizzle_bits(swz);
  return off ^ (((off >> 7) & ((1u << b) - 1u)) << 4);
}

// ---- device helpers (sm_90a PTX; translated by tools/ptxemu) ----------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Dynamic shared memory rounded up to a 1024-byte boundary (allocate 1024 extra bytes).
__device__ __forceinline__ unsigned char* align1024(unsigned char* p) {
  const uint32_t a = smem_u32(p);
  return p + (((a + 1023u) & ~1023u) - a);
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ uint64_t mbar_arrive(uint32_t bar) {
  uint64_t st;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(st) : "r"(bar) : "memory");
  return st;
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(tx)
               : "memory");
}

__device__ __forceinline__ uint32_t mbar_try_wait_parity(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok;
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait_parity(bar, parity)) {
  }
}

__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait_read0() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Data pattern helpers shared by host references and kernels.
__host__ __device__ inline uint32_t wg_hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
