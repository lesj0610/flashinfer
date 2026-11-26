// Host shim seen through the CUDA runtime (static cudart by default; the manifest also builds it
// with -cudart shared and with -DWGEMU_TEST_SPOOF=1). Every check goes through runtime API calls
// only, so it shows which driver entry points the runtime reaches the shim by:
//   * cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled") returns the shim's encoder, which
//     accepts a valid map and rejects boxDim[0] * elementSize % 16 != 0 (cuda.h 13.0);
//   * cudaLaunchKernelEx with a 1x1x1 cluster, programmatic stream serialization and a
//     programmatic event launches and the event completes; a 2x1x1 cluster fails with
//     cudaErrorInvalidClusterSize (912) and the kernel does not run;
//   * cluster function attributes and cluster occupancy queries answer for cluster size 1;
//   * a __cluster_dims__(1, 1, 1) kernel launches with <<<>>>;
//   * WGEMU_TEST_SPOOF=1: WGEMU_SPOOF_CC=1 is set before the first CUDA call; the device then
//     reports compute capability 9.0 and cluster launch support, and every launch above still
//     works [inferred: the runtime picks the kernel image by the device's real architecture].
#include <dlfcn.h>

#include <cstring>

#include "async_common.cuh"

#ifndef WGEMU_TEST_SPOOF
#define WGEMU_TEST_SPOOF 0
#endif

__global__ void k_add(int* p, int v) {
  if (threadIdx.x == 0) atomicAdd(p, v);
}

__global__ void __cluster_dims__(1, 1, 1) k_cluster1(int* p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
  if (threadIdx.x == 0) atomicAdd(p, (int)r);
}

static int read_int(const int* d) {
  int v = 0;
  WG_CHECK(cudaMemcpy(&v, d, sizeof(int), cudaMemcpyDeviceToHost));
  return v;
}

int main() {
#if WGEMU_TEST_SPOOF
  setenv("WGEMU_SPOOF_CC", "1", 1);
#endif
  const bool spoof = WGEMU_TEST_SPOOF != 0;
  {
    cudaDeviceProp prop;
    WG_CHECK(cudaGetDeviceProperties(&prop, 0));
    int cl = -1, major = -1;
    WG_CHECK(cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, 0));
    WG_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    Checker ck("async.shim.device_attributes");
    ck.eq(0, (unsigned)prop.major, spoof ? 9u : 8u);
    ck.eq(1, (unsigned)major, spoof ? 9u : 8u);
    if (spoof) ck.eq(2, (unsigned)prop.minor, 0u);
    ck.eq(3, (unsigned)cl, spoof ? 1u : 0u);
    wg_report(ck);
  }

  int* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 4 * sizeof(int)));
  WG_CHECK(cudaMemset(d, 0, 4 * sizeof(int)));

  {
    Checker ck("async.shim.encode_entry_point");
    const EncodeTiledFn enc = wg_encode_tiled();
    Dl_info info;
    std::memset(&info, 0, sizeof(info));
    const bool found = dladdr(reinterpret_cast<void*>(enc), &info) != 0 && info.dli_fname &&
                       std::strstr(info.dli_fname, "wgemu_shim") != nullptr;
    if (!found) std::printf("encoder resolved to %s\n", info.dli_fname ? info.dli_fname : "?");
    ck.ok(0, found);
    CUtensorMap tm;
    const cuuint64_t dims[2] = {64, 16};
    const cuuint64_t strides[1] = {128};
    const cuuint32_t es[2] = {1, 1};
    const cuuint32_t good[2] = {8, 4};  // 8 x u16 = 16 bytes
    const cuuint32_t bad[2] = {4, 4};   // 8 bytes: not a multiple of 16
    ck.eq(1,
          (unsigned)enc(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, d, dims, strides, good, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          (unsigned)CUDA_SUCCESS);
    ck.eq(2,
          (unsigned)enc(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, d, dims, strides, bad, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          (unsigned)CUDA_ERROR_INVALID_VALUE);
    wg_report(ck);
  }

  cudaStream_t s;
  cudaEvent_t ev;
  WG_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  WG_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  cudaLaunchAttribute at[3];
  std::memset(at, 0, sizeof(at));
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  at[2].id = cudaLaunchAttributeProgrammaticEvent;
  at[2].val.programmaticEvent.event = ev;
  at[2].val.programmaticEvent.flags = 0;
  at[2].val.programmaticEvent.triggerAtBlockStart = 0;
  cudaLaunchConfig_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3(4);
  cfg.blockDim = dim3(32);
  cfg.stream = s;
  cfg.attrs = at;
  cfg.numAttrs = 3;
  {
    Checker ck("async.shim.launch_ex_cluster1_pdl_event");
    ck.eq(0, (unsigned)cudaLaunchKernelEx(&cfg, k_add, d, 5), (unsigned)cudaSuccess);
    ck.eq(1, (unsigned)cudaEventSynchronize(ev), (unsigned)cudaSuccess);
    ck.eq(2, (unsigned)cudaEventQuery(ev), (unsigned)cudaSuccess);
    WG_CHECK(cudaStreamSynchronize(s));
    ck.eq(3, (unsigned)read_int(d), 20u);
    wg_report(ck);
  }
  {
    Checker ck("async.shim.launch_ex_cluster2_rejected");
    at[0].val.clusterDim.x = 2;
    cfg.numAttrs = 1;
    ck.eq(0, (unsigned)cudaLaunchKernelEx(&cfg, k_add, d, 100), 912u);
    (void)cudaGetLastError();
    WG_CHECK(cudaStreamSynchronize(s));
    ck.eq(1, (unsigned)read_int(d), 20u);
    wg_report(ck);
  }
  {
    Checker ck("async.shim.cluster_attributes_occupancy");
    ck.eq(0,
          (unsigned)cudaFuncSetAttribute(k_add, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
          (unsigned)cudaSuccess);
    cudaFuncAttributes fa;
    std::memset(&fa, 0xFF, sizeof(fa));
    ck.eq(1, (unsigned)cudaFuncGetAttributes(&fa, k_add), (unsigned)cudaSuccess);
    ck.eq(2, (unsigned)fa.requiredClusterWidth, 0u);
    ck.eq(3, (unsigned)fa.clusterDimMustBeSet, 0u);
    at[0].val.clusterDim.x = 1;
    cfg.numAttrs = 1;
    int nclusters = -1, csize = -1;
    ck.eq(4, (unsigned)cudaOccupancyMaxActiveClusters(&nclusters, k_add, &cfg),
          (unsigned)cudaSuccess);
    ck.ok(5, nclusters >= 1);
    cfg.numAttrs = 0;
    ck.eq(6, (unsigned)cudaOccupancyMaxPotentialClusterSize(&csize, k_add, &cfg),
          (unsigned)cudaSuccess);
    ck.eq(7, (unsigned)csize, 1u);
    (void)cudaGetLastError();
    wg_report(ck);
  }
  {
    Checker ck("async.shim.cluster_dims_kernel");
    k_cluster1<<<3, 32, 0, s>>>(d + 1);
    ck.eq(0, (unsigned)cudaGetLastError(), (unsigned)cudaSuccess);
    WG_CHECK(cudaStreamSynchronize(s));
    ck.eq(1, (unsigned)read_int(d + 1), 3u);  // %cluster_nctarank == 1 in each of 3 CTAs
    wg_report(ck);
  }

  WG_CHECK(cudaEventDestroy(ev));
  WG_CHECK(cudaStreamDestroy(s));
  WG_CHECK(cudaFree(d));
  return wg_finish();
}
