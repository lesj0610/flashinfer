/* Driver-level test of host/shim.c (owner A). Run with LD_PRELOAD of the shim, once plainly and
 * once as `WGEMU_SPOOF_CC=1 ... test_shim_driver spoof` (tests/async/host/run_host_tests.sh).
 *
 *   host.shim.encode_tiled        valid map (emulated format magic), every rejected argument class
 *   host.shim.encode_im2col       valid 3D / 4D maps, corner ranges, zero-area box, channel / pixel
 *                                 limits
 *   host.shim.replace_address     address rewrite, alignment, foreign map
 *   host.shim.proc_address        cuGetProcAddress(_v2) substitution: per-thread variant by flag,
 *                                 cuGetProcAddress variant by requested version, pass-through of
 *                                 other names, the legacy 4-argument entry
 *   host.shim.dlsym               dlsym on the libcuda handle and RTLD_DEFAULT; RTLD_NEXT from a
 *                                 library still resolves past it (libnext_a -> libnext_b)
 *   host.shim.launch_ex           1x1x1 cluster + programmatic serialization / event + memory sync
 *                                 domain + priority launch, event completion, 2x1x1 -> 912,
 *                                 0x1x1 -> invalid value, the _ptsz entry
 *   host.shim.func_attributes     cluster function / kernel attributes, pass-through of others
 *   host.shim.occupancy           cluster occupancy for cluster size 1, 2x1x1 -> 912
 *   host.shim.spoof               compute capability and cluster / tensor-map attributes, off or on
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wgmma_emu/tensormap.h"

#define CU(x)                                                                    \
  do {                                                                           \
    CUresult r_ = (x);                                                           \
    if (r_ != CUDA_SUCCESS) {                                                    \
      printf("driver error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

typedef struct {
  const char* name;
  long cases, mism;
} chk;

static int g_failed;

static void eq(chk* c, long idx, long long got, long long want) {
  ++c->cases;
  if (got != want) {
    if (c->mism < 8) {
      printf("WGEMU-MISMATCH name=%s idx=%ld got=%lld want=%lld\n", c->name, idx, got, want);
    }
    ++c->mism;
  }
}

static void report(const chk* c) {
  printf("WGEMU-TEST name=%s cases=%ld mismatches=%ld\n", c->name, c->cases, c->mism);
  if (c->mism || !c->cases) ++g_failed;
}

/* p is the exported function `sym` of the object whose path contains `lib`. */
static int is_fn(const void* p, const char* lib, const char* sym) {
  Dl_info i;
  memset(&i, 0, sizeof(i));
  if (!p || !dladdr(p, &i) || !i.dli_fname || !strstr(i.dli_fname, lib)) return 0;
  return sym == NULL || (i.dli_sname && strcmp(i.dli_sname, sym) == 0 && i.dli_saddr == p);
}

static const char k_ptx[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry k_add(.param .u64 k_add_p, .param .u32 k_add_v)\n"
    "{\n"
    "  .reg .b32 %r<3>;\n"
    "  .reg .b64 %rd<3>;\n"
    "  ld.param.u64 %rd1, [k_add_p];\n"
    "  ld.param.u32 %r1, [k_add_v];\n"
    "  cvta.to.global.u64 %rd2, %rd1;\n"
    "  atom.global.add.u32 %r2, [%rd2], %r1;\n"
    "  ret;\n"
    "}\n";

/* ---- tensor maps ------------------------------------------------------------------------- */
static CUresult tiled(void* tm, int dt, unsigned rank, void* base, const cuuint64_t* dims,
                      const cuuint64_t* strides, const cuuint32_t* box, const cuuint32_t* es,
                      int il, int swz, int oob) {
  return cuTensorMapEncodeTiled((CUtensorMap*)tm, (CUtensorMapDataType)dt, rank, base, dims,
                                strides, box, es, (CUtensorMapInterleave)il,
                                (CUtensorMapSwizzle)swz, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                (CUtensorMapFloatOOBfill)oob);
}

static void test_encode_tiled(char* base) {
  chk c = {"host.shim.encode_tiled", 0, 0};
  _Alignas(64) unsigned char raw[256];
  const int U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16, U32 = CU_TENSOR_MAP_DATA_TYPE_UINT32;
  const int INV = CUDA_ERROR_INVALID_VALUE;
  const cuuint64_t dims[6] = {64, 16, 2, 2, 2, 2};
  const cuuint64_t st[5] = {128, 2048, 4096, 8192, 16384};
  const cuuint64_t st_bad[1] = {100};
  const cuuint32_t box[6] = {8, 4, 1, 1, 1, 1}, box_odd[2] = {4, 4}, box_big[2] = {8, 257};
  const cuuint32_t box_wide[2] = {128, 4};
  const cuuint32_t es[6] = {1, 1, 1, 1, 1, 1}, es9[2] = {1, 9}, es0[2] = {1, 0};
  const cuuint32_t es_dim0[2] = {9, 1}; /* ignored for dimension 0 without interleave */
  eq(&c, 0, tiled(raw, U16, 2, base, dims, st, box, es, 0, 0, 0), CUDA_SUCCESS);
  uint32_t magic = 0;
  memcpy(&magic, raw + 12, 4);
  eq(&c, 1, magic, WGEMU_TMAP_MAGIC);
  eq(&c, 2, tiled(raw, U16, 2, base, dims, st, box, es_dim0, 0, 0, 0), CUDA_SUCCESS);
  eq(&c, 3, tiled(raw, U16, 5, base, dims, st, box, es, 0, 0, 0), CUDA_SUCCESS);
  eq(&c, 4, tiled(raw, U16, 0, base, dims, st, box, es, 0, 0, 0), INV);
  eq(&c, 5, tiled(raw, U16, 6, base, dims, st, box, es, 0, 0, 0), INV);
  eq(&c, 6, tiled(raw, U16, 2, base, dims, st, box_odd, es, 0, 0, 0), INV);
  eq(&c, 7, tiled(raw, U16, 2, base, dims, st, box_big, es, 0, 0, 0), INV);
  eq(&c, 8, tiled(raw, U16, 2, base, dims, st_bad, box, es, 0, 0, 0), INV);
  eq(&c, 9, tiled(raw, U16, 2, base, dims, st, box, es9, 0, 0, 0), INV);
  eq(&c, 10, tiled(raw, U16, 2, base, dims, st, box, es0, 0, 0, 0), INV);
  eq(&c, 11, tiled(raw + 8, U16, 2, base, dims, st, box, es, 0, 0, 0), INV);
  eq(&c, 12, tiled(raw, U16, 2, base + 8, dims, st, box, es, 0, 0, 0), INV);
  eq(&c, 13, tiled(raw, U16, 2, base, dims, st, box_wide, es, 0, CU_TENSOR_MAP_SWIZZLE_128B, 0),
     INV);
  eq(&c, 14,
     tiled(raw, U32, 2, base, dims, st, box, es, 0, 0,
           CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA),
     INV);
  eq(&c, 15, tiled(raw, 13, 2, base, dims, st, box, es, 0, 0, 0), INV);
  eq(&c, 16, tiled(raw, U16, 2, base, dims, st, box, es, 0, 4, 0), INV);
  eq(&c, 17, tiled(raw, U16, 2, base, dims, st, box, es, CU_TENSOR_MAP_INTERLEAVE_16B, 0, 0), INV);
  eq(&c, 18, tiled(NULL, U16, 2, base, dims, st, box, es, 0, 0, 0), INV);
  report(&c);
}

static CUresult im2col(void* tm, unsigned rank, void* base, const cuuint64_t* dims,
                       const cuuint64_t* strides, const int* lo, const int* hi, unsigned cpp,
                       unsigned ppc) {
  const cuuint32_t es[5] = {1, 1, 1, 1, 1};
  return cuTensorMapEncodeIm2col((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_UINT16, rank, base, dims,
                                 strides, lo, hi, cpp, ppc, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static void test_encode_im2col(char* base) {
  chk c = {"host.shim.encode_im2col", 0, 0};
  _Alignas(64) unsigned char raw[128];
  const int INV = CUDA_ERROR_INVALID_VALUE;
  const cuuint64_t d4[4] = {64, 9, 14, 2}, s4[3] = {128, 1152, 16128};
  const cuuint64_t d3[3] = {64, 400, 2}, s3[2] = {128, 51200};
  const int m1[2] = {-1, -1}, lo_big[2] = {128, 0}, lo_area[2] = {5, 0}, hi_area[2] = {-5, 0};
  const int lo3[1] = {-300}, hi3[1] = {-1}, lo3_bad[1] = {-32769};
  eq(&c, 0, im2col(raw, 4, base, d4, s4, m1, m1, 8, 64), CUDA_SUCCESS);
  uint32_t magic = 0;
  memcpy(&magic, raw + 12, 4);
  eq(&c, 1, magic, WGEMU_TMAP_MAGIC);
  eq(&c, 2, im2col(raw, 3, base, d3, s3, lo3, hi3, 16, 1024), CUDA_SUCCESS);
  eq(&c, 3, im2col(raw, 3, base, d3, s3, lo3_bad, hi3, 16, 32), INV);
  eq(&c, 4, im2col(raw, 2, base, d4, s4, m1, m1, 8, 64), INV);
  eq(&c, 5, im2col(raw, 4, base, d4, s4, lo_big, m1, 8, 64), INV);
  eq(&c, 6, im2col(raw, 4, base, d4, s4, lo_area, hi_area, 8, 64), INV);
  eq(&c, 7, im2col(raw, 4, base, d4, s4, m1, m1, 0, 64), INV);
  eq(&c, 8, im2col(raw, 4, base, d4, s4, m1, m1, 264, 64), INV);
  eq(&c, 9, im2col(raw, 4, base, d4, s4, m1, m1, 8, 0), INV);
  eq(&c, 10, im2col(raw, 4, base, d4, s4, m1, m1, 8, 1025), INV);
  eq(&c, 11, im2col(raw, 4, base, d4, s4, m1, m1, 4, 64), INV); /* 8-byte channel row */
  eq(&c, 12, im2col(raw, 4, base, d4, s4, NULL, m1, 8, 64), INV);
  report(&c);
}

static void test_replace_address(char* base) {
  chk c = {"host.shim.replace_address", 0, 0};
  _Alignas(64) unsigned char raw[128], foreign[128];
  const cuuint64_t dims[2] = {64, 16}, st[1] = {128};
  const cuuint32_t box[2] = {8, 4}, es[2] = {1, 1};
  CU(tiled(raw, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, base, dims, st, box, es, 0, 0, 0));
  eq(&c, 0, cuTensorMapReplaceAddress((CUtensorMap*)raw, base + 4096), CUDA_SUCCESS);
  uint64_t a = 0;
  memcpy(&a, raw, 8);
  eq(&c, 1, (long long)a, (long long)(uintptr_t)(base + 4096));
  eq(&c, 2, cuTensorMapReplaceAddress((CUtensorMap*)raw, base + 8), CUDA_ERROR_INVALID_VALUE);
  memset(foreign, 0, sizeof(foreign));
  eq(&c, 3, cuTensorMapReplaceAddress((CUtensorMap*)foreign, base), CUDA_ERROR_INVALID_VALUE);
  report(&c);
}

/* ---- entry points -------------------------------------------------------------------------- */
typedef CUresult (*legacy_gpa_fn)(const char*, void**, int, cuuint64_t);

static void* gpa(const char* sym, int version, cuuint64_t flags) {
  void* p = NULL;
  CUdriverProcAddressQueryResult st = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  if (cuGetProcAddress(sym, &p, version, flags, &st) != CUDA_SUCCESS ||
      st != CU_GET_PROC_ADDRESS_SUCCESS) {
    return NULL;
  }
  return p;
}

static void test_proc_address(void) {
  chk c = {"host.shim.proc_address", 0, 0};
  const cuuint64_t DEF = CU_GET_PROC_ADDRESS_DEFAULT;
  const cuuint64_t PT = CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
  const char* S = "wgemu_shim";
  eq(&c, 0, is_fn(gpa("cuTensorMapEncodeTiled", 12000, DEF), S, "cuTensorMapEncodeTiled"), 1);
  eq(&c, 1, is_fn(gpa("cuTensorMapEncodeIm2col", 12000, DEF), S, "cuTensorMapEncodeIm2col"), 1);
  eq(&c, 2, is_fn(gpa("cuTensorMapReplaceAddress", 12000, DEF), S, "cuTensorMapReplaceAddress"), 1);
  eq(&c, 3, is_fn(gpa("cuLaunchKernelEx", 12000, DEF), S, "cuLaunchKernelEx"), 1);
  eq(&c, 4, is_fn(gpa("cuLaunchKernelEx", 12000, PT), S, "cuLaunchKernelEx_ptsz"), 1);
  eq(&c, 5, is_fn(gpa("cuGetProcAddress", 12000, DEF), S, "cuGetProcAddress_v2"), 1);
  eq(&c, 6, is_fn(gpa("cuGetProcAddress", 11080, DEF), S, "cuGetProcAddress"), 1);
  eq(&c, 7, is_fn(gpa("cuDeviceGetAttribute", 12000, DEF), S, "cuDeviceGetAttribute"), 1);
  eq(&c, 8,
     is_fn(gpa("cuOccupancyMaxActiveClusters", 12000, DEF), S, "cuOccupancyMaxActiveClusters"), 1);
  eq(&c, 9, is_fn(gpa("cuMemAlloc", 12000, DEF), "libcuda", NULL), 1);
  eq(&c, 10, is_fn(gpa("cuStreamSynchronize", 12000, PT), "libcuda", NULL), 1);
  legacy_gpa_fn legacy = (legacy_gpa_fn)dlsym(RTLD_DEFAULT, "cuGetProcAddress");
  eq(&c, 11, is_fn((void*)legacy, S, "cuGetProcAddress"), 1);
  void* p = NULL;
  eq(&c, 12, legacy ? legacy("cuLaunchKernelEx", &p, 11080, DEF) : -1, CUDA_SUCCESS);
  eq(&c, 13, is_fn(p, S, "cuLaunchKernelEx"), 1);
  report(&c);
}

int wg_call_next(void); /* libnext_a */

static void test_dlsym(void) {
  chk c = {"host.shim.dlsym", 0, 0};
  const char* S = "wgemu_shim";
  void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  eq(&c, 0, h != NULL, 1);
  eq(&c, 1, is_fn(h ? dlsym(h, "cuTensorMapEncodeTiled") : NULL, S, "cuTensorMapEncodeTiled"), 1);
  eq(&c, 2, is_fn(h ? dlsym(h, "cuGetProcAddress_v2") : NULL, S, "cuGetProcAddress_v2"), 1);
  eq(&c, 3, is_fn(h ? dlsym(h, "cuLaunchKernelEx_ptsz") : NULL, S, "cuLaunchKernelEx_ptsz"), 1);
  eq(&c, 4, is_fn(h ? dlsym(h, "cuMemAlloc_v2") : NULL, "libcuda", NULL), 1);
  eq(&c, 5, is_fn(dlsym(RTLD_DEFAULT, "cuDeviceGetAttribute"), S, "cuDeviceGetAttribute"), 1);
  eq(&c, 6, wg_call_next(), 2);
  if (h) dlclose(h);
  report(&c);
}

/* ---- launches, attributes, occupancy ----------------------------------------------------------
 */
typedef CUresult (*launch_ex_fn)(const CUlaunchConfig*, CUfunction, void**, void**);

static unsigned read_u32(CUdeviceptr p) {
  unsigned v = 0;
  CU(cuMemcpyDtoH(&v, p, 4));
  return v;
}

static void test_launch(CUfunction f, CUdeviceptr out, CUstream s) {
  chk c = {"host.shim.launch_ex", 0, 0};
  CUevent ev;
  CU(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
  CU(cuMemsetD32(out, 0, 1));
  unsigned v = 5;
  void* args[2] = {&out, &v};
  CUlaunchAttribute at[5];
  memset(at, 0, sizeof(at));
  at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  at[0].value.clusterDim.x = at[0].value.clusterDim.y = at[0].value.clusterDim.z = 1;
  at[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  at[1].value.programmaticStreamSerializationAllowed = 1;
  at[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT;
  at[2].value.programmaticEvent.event = ev;
  at[2].value.programmaticEvent.flags = 0;
  at[2].value.programmaticEvent.triggerAtBlockStart = 0;
  at[3].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
  at[3].value.memSyncDomain = CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT;
  at[4].id = CU_LAUNCH_ATTRIBUTE_PRIORITY; /* kept: valid on SM80 */
  at[4].value.priority = 0;
  CUlaunchConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDimX = 4;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 1;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.hStream = s;
  cfg.attrs = at;
  cfg.numAttrs = 5;
  eq(&c, 0, cuLaunchKernelEx(&cfg, f, args, NULL), CUDA_SUCCESS);
  eq(&c, 1, cuEventSynchronize(ev), CUDA_SUCCESS);
  eq(&c, 2, cuEventQuery(ev), CUDA_SUCCESS);
  CU(cuStreamSynchronize(s));
  eq(&c, 3, read_u32(out), 20);
  at[0].value.clusterDim.x = 2;
  cfg.numAttrs = 1;
  eq(&c, 4, cuLaunchKernelEx(&cfg, f, args, NULL), CUDA_ERROR_INVALID_CLUSTER_SIZE);
  at[0].value.clusterDim.x = 0;
  eq(&c, 5, cuLaunchKernelEx(&cfg, f, args, NULL), CUDA_ERROR_INVALID_VALUE);
  CU(cuStreamSynchronize(s));
  eq(&c, 6, read_u32(out), 20);
  at[0].value.clusterDim.x = 1;
  launch_ex_fn ptsz = (launch_ex_fn)dlsym(RTLD_DEFAULT, "cuLaunchKernelEx_ptsz");
  eq(&c, 7, is_fn((void*)ptsz, "wgemu_shim", "cuLaunchKernelEx_ptsz"), 1);
  eq(&c, 8, ptsz ? ptsz(&cfg, f, args, NULL) : -1, CUDA_SUCCESS);
  CU(cuStreamSynchronize(s));
  eq(&c, 9, read_u32(out), 40);
  CU(cuEventDestroy(ev));
  report(&c);
}

static void test_func_attributes(CUfunction f, CUkernel k, CUdevice dev) {
  chk c = {"host.shim.func_attributes", 0, 0};
  int v = -1;
  eq(&c, 0, cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, f), CUDA_SUCCESS);
  eq(&c, 1, v, 0);
  v = -1;
  eq(&c, 2, cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, f), CUDA_SUCCESS);
  eq(&c, 3, v, 0);
  v = -1;
  eq(&c, 4, cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f), CUDA_SUCCESS);
  eq(&c, 5, v > 0, 1);
  eq(&c, 6, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, 2),
     CUDA_ERROR_INVALID_CLUSTER_SIZE);
  eq(&c, 7, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, 1), CUDA_SUCCESS);
  eq(&c, 8, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, -1),
     CUDA_ERROR_INVALID_VALUE);
  eq(&c, 9, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1),
     CUDA_SUCCESS);
  eq(&c, 10, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, 1),
     CUDA_SUCCESS);
  eq(&c, 11, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024),
     CUDA_SUCCESS);
  v = -1;
  eq(&c, 12, cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, k, dev),
     CUDA_SUCCESS);
  eq(&c, 13, v, 0);
  eq(&c, 14, cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, 3, k, dev),
     CUDA_ERROR_INVALID_CLUSTER_SIZE);
  eq(&c, 15, cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1, k, dev),
     CUDA_SUCCESS);
  report(&c);
}

static void test_occupancy(CUfunction f, CUdevice dev) {
  chk c = {"host.shim.occupancy", 0, 0};
  int bps = 0, sms = 0, n = -1, cs = -1;
  CU(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&bps, f, 128, 0, 0));
  CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CUlaunchAttribute at;
  memset(&at, 0, sizeof(at));
  at.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  at.value.clusterDim.x = at.value.clusterDim.y = at.value.clusterDim.z = 1;
  CUlaunchConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 128;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  eq(&c, 0, cuOccupancyMaxActiveClusters(&n, f, &cfg), CUDA_SUCCESS);
  eq(&c, 1, n, (long long)bps * sms);
  eq(&c, 2, n > 0, 1);
  eq(&c, 3, cuOccupancyMaxPotentialClusterSize(&cs, f, &cfg), CUDA_SUCCESS);
  eq(&c, 4, cs, 1);
  at.value.clusterDim.x = 2;
  eq(&c, 5, cuOccupancyMaxActiveClusters(&n, f, &cfg), CUDA_ERROR_INVALID_CLUSTER_SIZE);
  report(&c);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static void test_spoof(CUdevice dev, int spoof) {
  chk c = {"host.shim.spoof", 0, 0};
  int major = -1, minor = -1, cl = -1, tma = -1, ma = -1, mi = -1;
  CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CU(cuDeviceGetAttribute(&cl, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, dev));
  CU(cuDeviceGetAttribute(&tma, CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED, dev));
  CU(cuDeviceComputeCapability(&ma, &mi, dev));
  eq(&c, 0, major, spoof ? 9 : 8);
  if (spoof) eq(&c, 1, minor, 0);
  eq(&c, 2, cl, spoof ? 1 : 0);
  eq(&c, 3, tma, spoof ? 1 : 0);
  eq(&c, 4, ma, spoof ? 9 : 8);
  eq(&c, 5, mi, minor);
  /* The same answers through an entry point fetched with cuGetProcAddress. */
  typedef CUresult (*attr_fn)(int*, CUdevice_attribute, CUdevice);
  attr_fn a = (attr_fn)gpa("cuDeviceGetAttribute", 12000, CU_GET_PROC_ADDRESS_DEFAULT);
  int m2 = -1;
  eq(&c, 6, a ? a(&m2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) : -1, CUDA_SUCCESS);
  eq(&c, 7, m2, major);
  report(&c);
}
#pragma GCC diagnostic pop

int main(int argc, char** argv) {
  const int spoof = argc > 1 && strcmp(argv[1], "spoof") == 0;
  CUdevice dev;
  CUcontext ctx;
  CU(cuInit(0));
  CU(cuDeviceGet(&dev, 0));
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CUdeviceptr buf;
  CU(cuMemAlloc(&buf, 1u << 20));
  char* base = (char*)(uintptr_t)buf;

  test_encode_tiled(base);
  test_encode_im2col(base);
  test_replace_address(base);
  test_proc_address();
  test_dlsym();

  CUmodule mod;
  CUfunction f;
  CUlibrary lib;
  CUkernel k;
  CUstream s;
  CU(cuModuleLoadData(&mod, k_ptx));
  CU(cuModuleGetFunction(&f, mod, "k_add"));
  CU(cuLibraryLoadData(&lib, k_ptx, NULL, NULL, 0, NULL, NULL, 0));
  CU(cuLibraryGetKernel(&k, lib, "k_add"));
  CU(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  test_launch(f, buf, s);
  test_func_attributes(f, k, dev);
  test_occupancy(f, dev);
  test_spoof(dev, spoof);

  CU(cuStreamDestroy(s));
  CU(cuLibraryUnload(lib));
  CU(cuModuleUnload(mod));
  CU(cuMemFree(buf));
  CU(cuDevicePrimaryCtxRelease(dev));
  printf("WGEMU-TEST-END\n");
  return g_failed ? 1 : 0;
}
