/* Host shim of the SM80 wgmma emulation layer (fork-only). Owner: A. PLAN section 10.
 *
 *   gcc -O2 -shared -fPIC host/shim.c -ldl -o build/libwgemu_shim.so
 *   LD_PRELOAD=build/libwgemu_shim.so ./program
 *
 * - cuTensorMapEncodeTiled / cuTensorMapEncodeIm2col check their arguments against the driver
 *   documentation (cuda.h 13.0 comments of these functions) and write the emulated tensor-map
 *   format of include/wgmma_emu/tensormap.h. cuTensorMapReplaceAddress rewrites its address. The
 *   SM80 driver itself refuses to encode tensor maps (CUDA_ERROR_NOT_SUPPORTED).
 * - cuLaunchKernelEx(_ptsz) and cudaLaunchKernelExC(_ptsz) drop a 1x1x1 cluster dimension and the
 *   attributes that only act with clusters, programmatic dependent launch or memory synchronization
 *   domains; a cluster of more than one CTA fails with CUDA_ERROR_INVALID_CLUSTER_SIZE.
 * - Cluster function attributes and the cluster occupancy queries answer for cluster size 1.
 * - WGEMU_SPOOF_CC=1 reports compute capability 9.0, cluster launch and tensor-map access.
 * - Counting (E2): after each successful cuLaunchKernel(_ptsz), cuLaunchCooperativeKernel(_ptsz)
 *   or cuLaunchKernelEx(_ptsz) of a function whose module was translated with WGEMU_COUNT=1, the
 *   shim synchronizes the context, prints one "WGEMU-COUNT kernel=... grid=x,y,z ..." line on
 *   stdout and zeroes the module's counters. WGEMU_COUNT=0 at run time disables it.
 * - The shim's functions are returned on every path an application reaches the driver by: direct
 *   calls (exports below), cuGetProcAddress(_v2), cudaGetDriverEntryPoint(ByVersion)(_ptsz), and
 *   dlsym on the libcuda handle (static cudart resolves cuGetProcAddress that way).
 * Every dropped attribute or ignored setting is reported once on stderr; nothing is dropped
 * silently. The file depends on no CUDA header; the mirrored types are checked with static
 * asserts against the cuda.h 13.0 layouts.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/wgmma_emu/tensormap.h"

#define WG_EXPORT __attribute__((visibility("default")))

/* ---- mirrored driver / runtime types (cuda.h, driver_types.h of CUDA 13.0) ------------------ */
typedef int wg_result; /* CUresult and cudaError_t use the same codes for the values below */
enum {
  WG_SUCCESS = 0,
  WG_ERROR_INVALID_VALUE = 1,
  WG_ERROR_OUT_OF_MEMORY = 2, /* CUDA_ERROR_OUT_OF_MEMORY, cudaErrorMemoryAllocation */
  WG_ERROR_NOT_FOUND = 500,   /* CUDA_ERROR_NOT_FOUND, cudaErrorSymbolNotFound */
  WG_ERROR_INVALID_CLUSTER_SIZE = 912,
};

/* CUlaunchAttributeID / cudaLaunchAttributeID (identical numbering). */
enum {
  WG_ATTR_CLUSTER_DIMENSION = 4,
  WG_ATTR_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
  WG_ATTR_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
  WG_ATTR_PROGRAMMATIC_EVENT = 7,
  WG_ATTR_MEM_SYNC_DOMAIN_MAP = 9,
  WG_ATTR_MEM_SYNC_DOMAIN = 10,
  WG_ATTR_PREFERRED_CLUSTER_DIMENSION = 11,
};

/* CUfunction_attribute cluster entries. */
enum {
  WG_FUNC_CLUSTER_SIZE_MUST_BE_SET = 10,
  WG_FUNC_REQUIRED_CLUSTER_WIDTH = 11,
  WG_FUNC_REQUIRED_CLUSTER_HEIGHT = 12,
  WG_FUNC_REQUIRED_CLUSTER_DEPTH = 13,
  WG_FUNC_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
  WG_FUNC_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
};

/* CUdevice_attribute entries. */
enum {
  WG_DEV_MULTIPROCESSOR_COUNT = 16,
  WG_DEV_COMPUTE_CAPABILITY_MAJOR = 75,
  WG_DEV_COMPUTE_CAPABILITY_MINOR = 76,
  WG_DEV_CLUSTER_LAUNCH = 120,
  WG_DEV_TENSOR_MAP_ACCESS_SUPPORTED = 127,
};

#define WG_PROC_PER_THREAD_DEFAULT_STREAM 2u /* CU_GET_PROC_ADDRESS_*, cudaEnable* */
#define WG_PROC_LEGACY_STREAM 1u

typedef union wg_launch_attr_value { /* CUlaunchAttributeValue */
  char pad[64];
  struct {
    unsigned x, y, z;
  } cluster_dim;
  struct {
    void* event;
    int flags;
    int trigger_at_block_start;
  } programmatic_event;
  void* align_ptr;
} wg_launch_attr_value;

typedef struct wg_launch_attr { /* CUlaunchAttribute, cudaLaunchAttribute */
  int id;
  char pad[8 - sizeof(int)];
  wg_launch_attr_value value;
} wg_launch_attr;

typedef struct wg_cu_launch_config { /* CUlaunchConfig */
  unsigned grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes;
  void* stream;
  wg_launch_attr* attrs;
  unsigned num_attrs;
} wg_cu_launch_config;

typedef struct wg_dim3 {
  unsigned x, y, z;
} wg_dim3;

typedef struct wg_rt_launch_config { /* cudaLaunchConfig_t */
  wg_dim3 grid, block;
  size_t dynamic_smem_bytes;
  void* stream;
  wg_launch_attr* attrs;
  unsigned num_attrs;
} wg_rt_launch_config;

_Static_assert(sizeof(wg_launch_attr_value) == 64, "CUlaunchAttributeValue is 64 bytes");
_Static_assert(sizeof(wg_launch_attr) == 72, "CUlaunchAttribute is 72 bytes");
_Static_assert(offsetof(wg_launch_attr, value) == 8, "CUlaunchAttribute.value");
_Static_assert(offsetof(wg_cu_launch_config, stream) == 32, "CUlaunchConfig.hStream");
_Static_assert(offsetof(wg_cu_launch_config, attrs) == 40, "CUlaunchConfig.attrs");
_Static_assert(offsetof(wg_cu_launch_config, num_attrs) == 48, "CUlaunchConfig.numAttrs");
_Static_assert(offsetof(wg_rt_launch_config, dynamic_smem_bytes) == 24, "cudaLaunchConfig_t");
_Static_assert(offsetof(wg_rt_launch_config, stream) == 32, "cudaLaunchConfig_t.stream");
_Static_assert(offsetof(wg_rt_launch_config, attrs) == 40, "cudaLaunchConfig_t.attrs");
_Static_assert(offsetof(wg_rt_launch_config, num_attrs) == 48, "cudaLaunchConfig_t.numAttrs");

/* ---- messages ---------------------------------------------------------------------------- */
static unsigned long long g_noticed; /* one bit per notice kind */

static void notice_once(unsigned bit, const char* fmt, ...) {
  const unsigned long long m = 1ull << bit;
  if (__atomic_fetch_or(&g_noticed, m, __ATOMIC_RELAXED) & m) return;
  va_list ap;
  va_start(ap, fmt);
  fputs("wgemu-shim: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

static wg_result fail(wg_result code, const char* api, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "wgemu-shim: %s: ", api);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  return code;
}

/* ---- real symbols ------------------------------------------------------------------------ */
typedef void* (*dlsym_fn)(void*, const char*);
static dlsym_fn g_real_dlsym;

/* glibc's own dlsym (2.34 moved it into libc; older x86-64 / aarch64 versions as fallback). */
static dlsym_fn real_dlsym(void) {
  dlsym_fn f = __atomic_load_n(&g_real_dlsym, __ATOMIC_ACQUIRE);
  if (f) return f;
  static const char* const versions[] = {"GLIBC_2.34", "GLIBC_2.2.5", "GLIBC_2.17"};
  for (size_t i = 0; i < sizeof(versions) / sizeof(versions[0]) && !f; ++i) {
    f = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", versions[i]);
  }
  if (!f) {
    fputs("wgemu-shim: cannot find glibc dlsym\n", stderr);
    abort();
  }
  __atomic_store_n(&g_real_dlsym, f, __ATOMIC_RELEASE);
  return f;
}

static void* g_cuda; /* libcuda handle, kept open once found */

static void* keep_handle(void** slot, void* h) {
  void* expected = NULL;
  if (!__atomic_compare_exchange_n(slot, &expected, h, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    dlclose(h);
    return expected;
  }
  return h;
}

/* load = 0: only an already loaded libcuda (the dlsym hook must not load it). */
static void* cuda_handle(int load) {
  void* h = __atomic_load_n(&g_cuda, __ATOMIC_ACQUIRE);
  if (h) return h;
  h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!h && load) h = dlopen("libcuda.so.1", RTLD_NOW);
  return h ? keep_handle(&g_cuda, h) : NULL;
}

typedef struct real_sym {
  const char* name;
  void* p;
} real_sym;

static void* real_cuda(real_sym* s) {
  void* p = __atomic_load_n(&s->p, __ATOMIC_ACQUIRE);
  if (p) return p;
  void* h = cuda_handle(1);
  if (!h) return NULL;
  p = real_dlsym()(h, s->name);
  if (p) __atomic_store_n(&s->p, p, __ATOMIC_RELEASE);
  return p;
}

static void* g_cudart; /* cudart handle when it is not in the global scope */

/* cudart is either after the shim in the global scope or loaded locally (Python extensions);
 * a static cudart has no dynamic symbols and never reaches these wrappers through PLT. */
static void* real_cudart(real_sym* s) {
  void* p = __atomic_load_n(&s->p, __ATOMIC_ACQUIRE);
  if (p) return p;
  p = real_dlsym()(RTLD_NEXT, s->name);
  if (!p) {
    void* h = __atomic_load_n(&g_cudart, __ATOMIC_ACQUIRE);
    static const char* const sonames[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so"};
    for (size_t i = 0; i < sizeof(sonames) / sizeof(sonames[0]) && !h; ++i) {
      h = dlopen(sonames[i], RTLD_LAZY | RTLD_NOLOAD);
      if (h) h = keep_handle(&g_cudart, h);
    }
    if (h) p = real_dlsym()(h, s->name);
  }
  if (p) __atomic_store_n(&s->p, p, __ATOMIC_RELEASE);
  return p;
}

static wg_result missing(const char* name) {
  return fail(WG_ERROR_NOT_FOUND, name, "real symbol not found (is libcuda / libcudart loaded?)");
}

/* ---- tensor maps ------------------------------------------------------------------------- */
/* CUtensorMapDataType -> PTX .elemtype code (ptx.txt Table 36); -1 for unsupported. The packed
 * 16U4/16U6 types (13..15) carry sub-byte elements the device library does not move [inferred:
 * only used by sm_100 kernels]. */
static int ptx_elemtype(int dt) {
  static const signed char map[] = {
      WGEMU_TM_U8,      WGEMU_TM_U16,  WGEMU_TM_U32,     WGEMU_TM_S32, WGEMU_TM_U64,
      WGEMU_TM_S64,     WGEMU_TM_F16,  WGEMU_TM_F32,     WGEMU_TM_F64, WGEMU_TM_BF16,
      WGEMU_TM_F32_FTZ, WGEMU_TM_TF32, WGEMU_TM_TF32_FTZ};
  return (dt >= 0 && dt < (int)sizeof(map)) ? map[dt] : -1;
}

/* Checks shared by both encoders (cuda.h comments of cuTensorMapEncodeTiled / Im2col) and the
 * common fields of the emulated map. */
static wg_result encode_common(const char* api, void* tensor_map, int dtype, uint32_t rank,
                               uint32_t min_rank, void* gaddr, const uint64_t* gdim,
                               const uint64_t* gstride, const uint32_t* estride, int interleave,
                               int swizzle, int l2promo, int oob, wgemu_tmap* t, int* esize) {
  if (!tensor_map) return fail(WG_ERROR_INVALID_VALUE, api, "tensorMap is NULL");
  if ((uintptr_t)tensor_map & 63u) {
    return fail(WG_ERROR_INVALID_VALUE, api, "tensorMap %p is not 64-byte aligned", tensor_map);
  }
  if (dtype >= 13 && dtype <= 15) {
    return fail(WG_ERROR_INVALID_VALUE, api,
                "packed data type %d (16U4/16U6) is not supported by the SM80 emulation", dtype);
  }
  const int et = ptx_elemtype(dtype);
  if (et < 0) return fail(WG_ERROR_INVALID_VALUE, api, "tensorDataType %d is invalid", dtype);
  if (rank < min_rank || rank > 5) {
    return fail(WG_ERROR_INVALID_VALUE, api, "tensorRank %u outside [%u, 5]", rank, min_rank);
  }
  if (interleave < 0 || interleave > 2) {
    return fail(WG_ERROR_INVALID_VALUE, api, "interleave %d is invalid", interleave);
  }
  if (interleave != 0 && rank < 3) {
    return fail(WG_ERROR_INVALID_VALUE, api, "interleaved layouts need tensorRank >= 3");
  }
  if (swizzle >= 4 && swizzle <= 6) {
    return fail(WG_ERROR_INVALID_VALUE, api,
                "swizzle %d (128B atom 32B/64B) is not supported by the SM80 emulation", swizzle);
  }
  if (swizzle < 0 || swizzle > 6) {
    return fail(WG_ERROR_INVALID_VALUE, api, "swizzle %d is invalid", swizzle);
  }
  if (interleave == 2 && swizzle != 1) {
    return fail(WG_ERROR_INVALID_VALUE, api, "INTERLEAVE_32B requires SWIZZLE_32B");
  }
  if (l2promo < 0 || l2promo > 3) {
    return fail(WG_ERROR_INVALID_VALUE, api, "l2Promotion %d is invalid", l2promo);
  }
  if (oob < 0 || oob > 1) return fail(WG_ERROR_INVALID_VALUE, api, "oobFill %d is invalid", oob);
  if (oob == 1 && !wgemu_tm_elem_is_float(et)) {
    return fail(WG_ERROR_INVALID_VALUE, api, "NaN OOB fill needs a floating-point data type");
  }
  const uint64_t align = interleave == 2 ? 32u : 16u;
  if ((uintptr_t)gaddr % align) {
    return fail(WG_ERROR_INVALID_VALUE, api, "globalAddress %p is not %u-byte aligned", gaddr,
                (unsigned)align);
  }
  if (!gdim || (rank > 1 && !gstride) || !estride) {
    return fail(WG_ERROR_INVALID_VALUE, api, "NULL globalDim / globalStrides / elementStrides");
  }
  for (uint32_t i = 0; i < rank; ++i) {
    if (gdim[i] == 0 || gdim[i] > (1ull << 32)) {
      return fail(WG_ERROR_INVALID_VALUE, api, "globalDim[%u] = %llu outside [1, 2^32]", i,
                  (unsigned long long)gdim[i]);
    }
  }
  for (uint32_t i = 0; i + 1 < rank; ++i) {
    if (gstride[i] % align || gstride[i] >= (1ull << 40)) {
      return fail(WG_ERROR_INVALID_VALUE, api,
                  "globalStrides[%u] = %llu not a multiple of %u below 2^40", i,
                  (unsigned long long)gstride[i], (unsigned)align);
    }
  }
  for (uint32_t i = 0; i < rank; ++i) {
    if (i == 0 && interleave == 0) continue; /* ignored for dimension 0 (cuda.h) */
    if (estride[i] == 0 || estride[i] > 8) {
      return fail(WG_ERROR_INVALID_VALUE, api, "elementStrides[%u] = %u outside [1, 8]", i,
                  estride[i]);
    }
  }

  memset(t, 0, sizeof(*t));
  t->global_address = (uint64_t)(uintptr_t)gaddr;
  t->magic = WGEMU_TMAP_MAGIC;
  uint32_t misc = 0;
  misc = wgemu_tm_set(misc, WGEMU_TM_RANK_M1_SHIFT, 3, rank - 1);
  misc = wgemu_tm_set(misc, WGEMU_TM_ELEMTYPE_SHIFT, 4, (uint32_t)et);
  misc = wgemu_tm_set(misc, WGEMU_TM_INTERLEAVE_SHIFT, 2, (uint32_t)interleave);
  misc = wgemu_tm_set(misc, WGEMU_TM_SWIZZLE_SHIFT, 3, (uint32_t)swizzle); /* 0..3 = PTX codes */
  misc = wgemu_tm_set(misc, WGEMU_TM_ATOMICITY_SHIFT, 2, WGEMU_TM_ATOM_16B);
  misc = wgemu_tm_set(misc, WGEMU_TM_FILL_SHIFT, 1, (uint32_t)oob);
  misc = wgemu_tm_set(misc, WGEMU_TM_L2PROMO_SHIFT, 2, (uint32_t)l2promo);
  t->misc = misc;
  for (uint32_t i = 0; i < rank; ++i) {
    t->global_dim_m1[i] = (uint32_t)(gdim[i] - 1u);
    t->element_stride_m1[i] = (i == 0 && interleave == 0) ? 0u : (uint8_t)(estride[i] - 1u);
  }
  for (uint32_t i = 0; i + 1 < rank; ++i) t->global_stride[i] = gstride[i];
  *esize = wgemu_tm_elem_bytes(et);
  return WG_SUCCESS;
}

/* Inner bounding-box bytes against the swizzle span (cuda.h: interleave NONE with a swizzle). */
static wg_result check_inner(const char* api, int interleave, int swizzle, uint64_t inner) {
  const int span = wgemu_tm_swizzle_bytes(swizzle);
  if (interleave == 0 && span != 0 && inner > (uint64_t)span) {
    return fail(WG_ERROR_INVALID_VALUE, api, "inner box %llu bytes exceeds the %d-byte swizzle",
                (unsigned long long)inner, span);
  }
  return WG_SUCCESS;
}

WG_EXPORT wg_result cuTensorMapEncodeTiled(void* tensor_map, int dtype, uint32_t rank, void* gaddr,
                                           const uint64_t* gdim, const uint64_t* gstride,
                                           const uint32_t* box, const uint32_t* estride,
                                           int interleave, int swizzle, int l2promo, int oob) {
  static const char api[] = "cuTensorMapEncodeTiled";
  wgemu_tmap t;
  int esize = 0;
  wg_result r = encode_common(api, tensor_map, dtype, rank, 1, gaddr, gdim, gstride, estride,
                              interleave, swizzle, l2promo, oob, &t, &esize);
  if (r) return r;
  if (!box) return fail(WG_ERROR_INVALID_VALUE, api, "boxDim is NULL");
  for (uint32_t i = 0; i < rank; ++i) {
    if (box[i] == 0 || box[i] > 256) {
      return fail(WG_ERROR_INVALID_VALUE, api, "boxDim[%u] = %u outside [1, 256]", i, box[i]);
    }
    t.box_dim_m1[i] = (uint8_t)(box[i] - 1u);
  }
  const uint64_t inner = (uint64_t)box[0] * (uint64_t)esize;
  if (interleave == 0 && inner % 16u) {
    return fail(WG_ERROR_INVALID_VALUE, api, "boxDim[0] * element size = %llu not a multiple of 16",
                (unsigned long long)inner);
  }
  r = check_inner(api, interleave, swizzle, inner);
  if (r) return r;
  memcpy(tensor_map, &t, sizeof(t));
  return WG_SUCCESS;
}

/* Corner ranges per rank (cuda.h cuTensorMapEncodeIm2col; ptx.txt 3302-3318). */
static const int k_corner_lim[6] = {0, 0, 0, 32768, 128, 16};

WG_EXPORT wg_result cuTensorMapEncodeIm2col(void* tensor_map, int dtype, uint32_t rank, void* gaddr,
                                            const uint64_t* gdim, const uint64_t* gstride,
                                            const int* lower, const int* upper, uint32_t channels,
                                            uint32_t pixels, const uint32_t* estride,
                                            int interleave, int swizzle, int l2promo, int oob) {
  static const char api[] = "cuTensorMapEncodeIm2col";
  wgemu_tmap t;
  int esize = 0;
  wg_result r = encode_common(api, tensor_map, dtype, rank, 3, gaddr, gdim, gstride, estride,
                              interleave, swizzle, l2promo, oob, &t, &esize);
  if (r) return r;
  if (!lower || !upper) return fail(WG_ERROR_INVALID_VALUE, api, "NULL pixel box corner");
  const int lim = k_corner_lim[rank];
  for (uint32_t s = 0; s + 2 < rank; ++s) {
    if (lower[s] < -lim || lower[s] > lim - 1 || upper[s] < -lim || upper[s] > lim - 1) {
      return fail(WG_ERROR_INVALID_VALUE, api, "corner %u (%d, %d) outside [%d, %d]", s, lower[s],
                  upper[s], -lim, lim - 1);
    }
    /* Non-zero area: the last position dim - 1 + upper is not before lower [inferred]. */
    if ((int64_t)gdim[1 + s] - 1 + upper[s] < (int64_t)lower[s]) {
      return fail(WG_ERROR_INVALID_VALUE, api, "pixel box of dimension %u has zero area", s + 1);
    }
    t.lower_corner[s] = (int16_t)lower[s];
    t.upper_corner[s] = (int16_t)upper[s];
  }
  if (channels == 0 || channels > 256) {
    return fail(WG_ERROR_INVALID_VALUE, api, "channelsPerPixel %u outside [1, 256]", channels);
  }
  if (pixels == 0 || pixels > 1024) {
    return fail(WG_ERROR_INVALID_VALUE, api, "pixelsPerColumn %u outside [1, 1024]", pixels);
  }
  const uint64_t inner = (uint64_t)channels * (uint64_t)esize;
  /* The bounding box is a multiple of 16 bytes (ptx.txt 3214-3215); applied to the channel row
   * as cuda.h applies it to boxDim[0] in tiled mode [inferred]. */
  if (interleave == 0 && inner % 16u) {
    return fail(WG_ERROR_INVALID_VALUE, api,
                "channelsPerPixel * element size = %llu not a multiple of 16",
                (unsigned long long)inner);
  }
  r = check_inner(api, interleave, swizzle, inner);
  if (r) return r;
  t.misc = wgemu_tm_set(t.misc, WGEMU_TM_IM2COL_SHIFT, 1, 1u);
  t.channels_m1 = (uint8_t)(channels - 1u);
  t.pixels_m1 = (uint16_t)(pixels - 1u);
  memcpy(tensor_map, &t, sizeof(t));
  return WG_SUCCESS;
}

WG_EXPORT wg_result cuTensorMapReplaceAddress(void* tensor_map, void* gaddr) {
  static const char api[] = "cuTensorMapReplaceAddress";
  if (!tensor_map || ((uintptr_t)tensor_map & 63u)) {
    return fail(WG_ERROR_INVALID_VALUE, api, "tensorMap %p is NULL or not 64-byte aligned",
                tensor_map);
  }
  wgemu_tmap t;
  memcpy(&t, tensor_map, sizeof(t));
  if (t.magic != WGEMU_TMAP_MAGIC) {
    return fail(WG_ERROR_INVALID_VALUE, api, "tensorMap was not encoded by the wgemu shim");
  }
  const uintptr_t align = wgemu_tm_interleave(&t) == WGEMU_TM_INTERLEAVE_32B ? 32u : 16u;
  if ((uintptr_t)gaddr % align) {
    return fail(WG_ERROR_INVALID_VALUE, api, "globalAddress %p is not %u-byte aligned", gaddr,
                (unsigned)align);
  }
  t.global_address = (uint64_t)(uintptr_t)gaddr;
  memcpy(tensor_map, &t, sizeof(t));
  return WG_SUCCESS;
}

/* ---- launch attributes ------------------------------------------------------------------- */
#define WG_STACK_ATTRS 16

typedef struct wg_filtered {
  wg_launch_attr* attrs; /* NULL: nothing dropped, launch with the caller's list */
  unsigned n;
  int has_event;
  void* event;
  int event_flags;
  wg_launch_attr stack[WG_STACK_ATTRS];
} wg_filtered;

static const char* attr_reason(int id) {
  switch (id) {
    case WG_ATTR_CLUSTER_DIMENSION:
      return "cluster dimension 1x1x1 (cluster size 1 is the plain launch)";
    case WG_ATTR_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
      return "cluster scheduling policy (no clusters on SM80)";
    case WG_ATTR_PROGRAMMATIC_STREAM_SERIALIZATION:
      return "programmatic stream serialization (the launch waits for the previous kernel)";
    case WG_ATTR_PROGRAMMATIC_EVENT:
      return "programmatic event (recorded after the launch instead)";
    case WG_ATTR_MEM_SYNC_DOMAIN_MAP:
      return "memory sync domain map (one domain: fences cover more, never less)";
    case WG_ATTR_MEM_SYNC_DOMAIN:
      return "memory sync domain (one domain: fences cover more, never less)";
    case WG_ATTR_PREFERRED_CLUSTER_DIMENSION:
      return "preferred cluster dimension (no clusters on SM80)";
    default:
      return "";
  }
}

static void free_filtered(wg_filtered* f) {
  if (f->attrs && f->attrs != f->stack) free(f->attrs);
}

/* Builds the attribute list the SM80 driver accepts. Attribute 12 (launch completion event) and
 * every other attribute are kept: they are valid below compute capability 9.0 [inferred for the
 * ones cuda.h does not mark]. */
static wg_result filter_attrs(const char* api, const wg_launch_attr* in, unsigned n,
                              wg_filtered* f) {
  f->attrs = NULL;
  f->n = n;
  f->has_event = 0;
  if (!in || n == 0) return WG_SUCCESS;
  unsigned drop = 0;
  for (unsigned i = 0; i < n; ++i) {
    const int id = in[i].id;
    if (id == WG_ATTR_CLUSTER_DIMENSION) {
      const unsigned x = in[i].value.cluster_dim.x, y = in[i].value.cluster_dim.y,
                     z = in[i].value.cluster_dim.z;
      if (x == 0 || y == 0 || z == 0) {
        return fail(WG_ERROR_INVALID_VALUE, api, "cluster dimension %ux%ux%u has a zero", x, y, z);
      }
      if (x != 1 || y != 1 || z != 1) {
        return fail(WG_ERROR_INVALID_CLUSTER_SIZE, api,
                    "cluster dimension %ux%ux%u: the SM80 emulation runs cluster size 1 only", x, y,
                    z);
      }
      ++drop;
    } else if (id == WG_ATTR_CLUSTER_SCHEDULING_POLICY_PREFERENCE ||
               id == WG_ATTR_PROGRAMMATIC_STREAM_SERIALIZATION ||
               id == WG_ATTR_PROGRAMMATIC_EVENT || id == WG_ATTR_MEM_SYNC_DOMAIN_MAP ||
               id == WG_ATTR_MEM_SYNC_DOMAIN || id == WG_ATTR_PREFERRED_CLUSTER_DIMENSION) {
      ++drop;
    }
  }
  if (drop == 0) return WG_SUCCESS;
  wg_launch_attr* out = f->stack;
  if (n > WG_STACK_ATTRS) {
    out = (wg_launch_attr*)malloc(n * sizeof(wg_launch_attr));
    if (!out) return fail(WG_ERROR_OUT_OF_MEMORY, api, "out of host memory");
  }
  unsigned k = 0;
  for (unsigned i = 0; i < n; ++i) {
    const int id = in[i].id;
    const char* why = attr_reason(id);
    if (why[0] == '\0') {
      out[k++] = in[i];
      continue;
    }
    notice_once((unsigned)id, "dropped launch attribute %d: %s", id, why);
    if (id == WG_ATTR_PROGRAMMATIC_EVENT) {
      f->has_event = 1;
      f->event = in[i].value.programmatic_event.event;
      f->event_flags = in[i].value.programmatic_event.flags;
    }
  }
  f->attrs = out;
  f->n = k;
  return WG_SUCCESS;
}

/* ---- E2 counters (WGEMU_COUNT=1 at translation) ------------------------------------------ */
/* A module translated with --count holds the block WG_CNT_SYMBOL (layout: "E2 counters" in
 * include/wgmma_emu/async_device.cuh). After every successful launch of a function whose module
 * has it, the shim waits for the context, prints one WGEMU-COUNT line on stdout and zeroes the
 * block. WGEMU_COUNT=0 at run time turns this off. Launches during stream capture are not
 * counted (their counts go to the next counted launch of the module; reported once). */
#define WG_CNT_SYMBOL "__wgemu_global_count_v1"
#define WG_CNT_WORDS 136u     /* WGEMU_CNT_BYTES / 8 */
#define WG_CNT_WGMMA_TABLE 8u /* word index of WGEMU_CNT_WGMMA_TABLE */
#define WG_CNT_WGMMA_SLOTS 64u
#define WG_CACHE_SLOTS 4096u

typedef wg_result (*func_get_module_fn)(void**, void*);
typedef wg_result (*get_name_fn)(const char**, void*);
typedef wg_result (*module_get_global_fn)(unsigned long long*, size_t*, void*, const char*);
typedef wg_result (*kernel_get_function_fn)(void**, void*);
typedef wg_result (*kernel_get_library_fn)(void**, void*);
typedef wg_result (*ctx_sync_fn)(void);
typedef wg_result (*stream_is_capturing_fn)(void*, int*);
typedef wg_result (*memcpy_dtoh_fn)(void*, unsigned long long, size_t);
typedef wg_result (*memset_d8_fn)(unsigned long long, unsigned char, size_t);

static real_sym r_func_get_module = {"cuFuncGetModule", NULL};
static real_sym r_func_get_name = {"cuFuncGetName", NULL};
static real_sym r_module_get_global = {"cuModuleGetGlobal_v2", NULL};
static real_sym r_kernel_get_function = {"cuKernelGetFunction", NULL};
static real_sym r_kernel_get_library = {"cuKernelGetLibrary", NULL};
static real_sym r_kernel_get_name = {"cuKernelGetName", NULL};
static real_sym r_library_get_global = {"cuLibraryGetGlobal", NULL};
static real_sym r_ctx_sync = {"cuCtxSynchronize", NULL};
static real_sym r_stream_is_capturing = {"cuStreamIsCapturing", NULL};
static real_sym r_stream_is_capturing_ptsz = {"cuStreamIsCapturing_ptsz", NULL};
static real_sym r_memcpy_dtoh = {"cuMemcpyDtoH_v2", NULL};
static real_sym r_memset_d8 = {"cuMemsetD8_v2", NULL};

typedef struct wg_count_entry {
  void* f;                 /* launched handle (CUfunction or CUkernel) */
  unsigned long long dptr; /* counter block, 0: the module is not counted */
  char* name;
} wg_count_entry;

static wg_count_entry g_count_cache[WG_CACHE_SLOTS];
static unsigned g_count_cached;
static int g_count_lock;

static void count_lock(void) {
  while (__atomic_test_and_set(&g_count_lock, __ATOMIC_ACQUIRE)) {
  }
}

static void count_unlock(void) { __atomic_clear(&g_count_lock, __ATOMIC_RELEASE); }

static int count_enabled(void) {
  const char* e = getenv("WGEMU_COUNT");
  return !(e && strcmp(e, "0") == 0);
}

/* Counter block and kernel name of the launched handle; CUkernel handles (cudart >= 12 launches
 * through cuLibrary objects) go through cuKernelGetFunction / cuKernelGetLibrary. */
static void count_lookup(void* f, wg_count_entry* e) {
  func_get_module_fn get_mod = (func_get_module_fn)real_cuda(&r_func_get_module);
  get_name_fn func_name = (get_name_fn)real_cuda(&r_func_get_name);
  get_name_fn kernel_name = (get_name_fn)real_cuda(&r_kernel_get_name);
  module_get_global_fn mod_global = (module_get_global_fn)real_cuda(&r_module_get_global);
  kernel_get_function_fn kfunc = (kernel_get_function_fn)real_cuda(&r_kernel_get_function);
  kernel_get_library_fn klib = (kernel_get_library_fn)real_cuda(&r_kernel_get_library);
  module_get_global_fn lib_global = (module_get_global_fn)real_cuda(&r_library_get_global);
  void* func = f;
  void* mod = NULL;
  size_t bytes = 0;
  const char* name = NULL;
  e->f = f;
  e->dptr = 0;
  e->name = NULL;
  if (!get_mod || !mod_global) return;
  if (get_mod(&mod, f) != WG_SUCCESS) {
    mod = NULL;
    void* f2 = NULL;
    if (kfunc && kfunc(&f2, f) == WG_SUCCESS && f2 && get_mod(&mod, f2) == WG_SUCCESS) {
      func = f2;
    } else {
      mod = NULL;
    }
  }
  if (mod) {
    if (mod_global(&e->dptr, &bytes, mod, WG_CNT_SYMBOL) != WG_SUCCESS) e->dptr = 0;
  } else if (klib && lib_global) {
    void* lib = NULL;
    if (klib(&lib, f) != WG_SUCCESS ||
        lib_global(&e->dptr, &bytes, lib, WG_CNT_SYMBOL) != WG_SUCCESS) {
      e->dptr = 0;
    }
  }
  if (e->dptr && bytes < WG_CNT_WORDS * 8u) {
    notice_once(44u, "%s is %zu bytes (expected %u); launch not counted", WG_CNT_SYMBOL, bytes,
                WG_CNT_WORDS * 8u);
    e->dptr = 0;
  }
  if (!e->dptr) return;
  if (!(func_name && func_name(&name, func) == WG_SUCCESS && name) &&
      !(kernel_name && kernel_name(&name, f) == WG_SUCCESS && name)) {
    name = "?";
  }
  e->name = strdup(name);
}

/* Decoding of the wgmma code (keep in sync with tools/ptxemu/instrument.py WGMMA_TYPES). */
static const char* const k_wgmma_types[16] = {"f16", "bf16", "tf32", "f32", "e4m3", "e5m2",
                                              "s8",  "u8",   "s32",  "b1",  "t10",  "t11",
                                              "t12", "t13",  "t14",  "t15"};

static void count_print(const char* name, unsigned gx, unsigned gy, unsigned gz,
                        const unsigned long long* c) {
  unsigned order[WG_CNT_WGMMA_SLOTS];
  unsigned n = 0;
  for (unsigned i = 0; i < WG_CNT_WGMMA_SLOTS; ++i) {
    const unsigned code = (unsigned)(c[WG_CNT_WGMMA_TABLE + 2u * i] & 0xFFFFFFFFull);
    if (!code) continue;
    unsigned k = n++;
    while (k > 0) { /* sort by code: stable output order */
      const unsigned prev = (unsigned)(c[WG_CNT_WGMMA_TABLE + 2u * order[k - 1]] & 0xFFFFFFFFull);
      if (prev <= code) break;
      order[k] = order[k - 1];
      --k;
    }
    order[k] = i;
  }
  printf("WGEMU-COUNT kernel=%s grid=%u,%u,%u", name, gx, gy, gz);
  for (unsigned j = 0; j < n; ++j) {
    const unsigned i = order[j];
    const unsigned code = (unsigned)(c[WG_CNT_WGMMA_TABLE + 2u * i] & 0xFFFFFFFFull);
    printf(" wgmma[%sm64n%uk%u.%s.%s.%s]=%llu", (code & 2u) ? "sp." : "", (code >> 2) & 0x1FFu,
           (code >> 11) & 0x1FFu, k_wgmma_types[(code >> 20) & 15u],
           k_wgmma_types[(code >> 24) & 15u], k_wgmma_types[(code >> 28) & 15u],
           c[WG_CNT_WGMMA_TABLE + 2u * i + 1u]);
  }
  if (c[5]) printf(" wgmma_overflow=%llu", c[5]);
  printf(" bulk_bytes=%llu tma_bytes=%llu cpasync_bytes=%llu ldg_bytes=%llu mbar_waits=%llu\n",
         c[0], c[1], c[2], c[3], c[4]);
  fflush(stdout);
}

static void count_after(int ptsz, void* f, unsigned gx, unsigned gy, unsigned gz, void* stream) {
  if (!f || !count_enabled()) return;
  wg_count_entry e;
  int found = 0;
  count_lock();
  for (unsigned i = 0; i < g_count_cached; ++i) {
    if (g_count_cache[i].f == f) {
      e = g_count_cache[i];
      found = 1;
      break;
    }
  }
  count_unlock();
  if (!found) {
    count_lookup(f, &e);
    count_lock();
    if (g_count_cached < WG_CACHE_SLOTS) {
      g_count_cache[g_count_cached++] = e;
    } else {
      notice_once(45u, "counter lookup cache full; lookups repeat for further kernels");
    }
    count_unlock();
  }
  if (!e.dptr) return;
  stream_is_capturing_fn cap = (stream_is_capturing_fn)real_cuda(ptsz ? &r_stream_is_capturing_ptsz
                                                                      : &r_stream_is_capturing);
  int status = 0;
  if (cap && cap(stream, &status) == WG_SUCCESS && status != 0) {
    notice_once(46u, "launch of %s during stream capture not counted", e.name);
    return;
  }
  ctx_sync_fn sync = (ctx_sync_fn)real_cuda(&r_ctx_sync);
  memcpy_dtoh_fn dtoh = (memcpy_dtoh_fn)real_cuda(&r_memcpy_dtoh);
  memset_d8_fn zero = (memset_d8_fn)real_cuda(&r_memset_d8);
  unsigned long long c[WG_CNT_WORDS];
  if (!sync || !dtoh || !zero || sync() != WG_SUCCESS || dtoh(c, e.dptr, sizeof(c)) != WG_SUCCESS) {
    fprintf(stderr, "wgemu-shim: WGEMU-COUNT kernel=%s: reading the counters failed\n", e.name);
    return;
  }
  count_print(e.name, gx, gy, gz, c);
  if (zero(e.dptr, 0, sizeof(c)) != WG_SUCCESS || sync() != WG_SUCCESS) {
    fprintf(stderr, "wgemu-shim: WGEMU-COUNT kernel=%s: zeroing the counters failed\n", e.name);
  }
}

typedef wg_result (*cu_launch_fn)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  unsigned, void*, void**, void**);
typedef wg_result (*cu_launch_coop_fn)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, void*, void**);

static real_sym r_launch = {"cuLaunchKernel", NULL};
static real_sym r_launch_ptsz = {"cuLaunchKernel_ptsz", NULL};
static real_sym r_launch_coop = {"cuLaunchCooperativeKernel", NULL};
static real_sym r_launch_coop_ptsz = {"cuLaunchCooperativeKernel_ptsz", NULL};

static wg_result launch(int ptsz, void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                        unsigned by, unsigned bz, unsigned smem, void* stream, void** params,
                        void** extra) {
  cu_launch_fn real = (cu_launch_fn)real_cuda(ptsz ? &r_launch_ptsz : &r_launch);
  if (!real) return missing(ptsz ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
  wg_result r = real(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
  if (r == WG_SUCCESS) count_after(ptsz, f, gx, gy, gz, stream);
  return r;
}

WG_EXPORT wg_result cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                   unsigned by, unsigned bz, unsigned smem, void* stream,
                                   void** params, void** extra) {
  return launch(0, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
}

WG_EXPORT wg_result cuLaunchKernel_ptsz(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                        unsigned by, unsigned bz, unsigned smem, void* stream,
                                        void** params, void** extra) {
  return launch(1, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
}

static wg_result launch_coop(int ptsz, void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                             unsigned by, unsigned bz, unsigned smem, void* stream, void** params) {
  cu_launch_coop_fn real =
      (cu_launch_coop_fn)real_cuda(ptsz ? &r_launch_coop_ptsz : &r_launch_coop);
  if (!real) return missing(ptsz ? "cuLaunchCooperativeKernel_ptsz" : "cuLaunchCooperativeKernel");
  wg_result r = real(f, gx, gy, gz, bx, by, bz, smem, stream, params);
  if (r == WG_SUCCESS) count_after(ptsz, f, gx, gy, gz, stream);
  return r;
}

WG_EXPORT wg_result cuLaunchCooperativeKernel(void* f, unsigned gx, unsigned gy, unsigned gz,
                                              unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                              void* stream, void** params) {
  return launch_coop(0, f, gx, gy, gz, bx, by, bz, smem, stream, params);
}

WG_EXPORT wg_result cuLaunchCooperativeKernel_ptsz(void* f, unsigned gx, unsigned gy, unsigned gz,
                                                   unsigned bx, unsigned by, unsigned bz,
                                                   unsigned smem, void* stream, void** params) {
  return launch_coop(1, f, gx, gy, gz, bx, by, bz, smem, stream, params);
}

typedef wg_result (*cu_launch_ex_fn)(const wg_cu_launch_config*, void*, void**, void**);
typedef wg_result (*event_record_fn)(void*, void*, unsigned);

static real_sym r_launch_ex = {"cuLaunchKernelEx", NULL};
static real_sym r_launch_ex_ptsz = {"cuLaunchKernelEx_ptsz", NULL};
static real_sym r_event_record = {"cuEventRecordWithFlags", NULL};
static real_sym r_event_record_ptsz = {"cuEventRecordWithFlags_ptsz", NULL};

/* The programmatic event triggers once every block triggered or exited; recording it on the
 * stream after the kernel fires it no earlier than that [inferred]. */
static wg_result launch_ex(int ptsz, const wg_cu_launch_config* cfg, void* f, void** params,
                           void** extra) {
  const char* api = ptsz ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx";
  cu_launch_ex_fn real = (cu_launch_ex_fn)real_cuda(ptsz ? &r_launch_ex_ptsz : &r_launch_ex);
  if (!real) return missing(api);
  if (!cfg) return real(cfg, f, params, extra);
  wg_filtered flt;
  wg_result r = filter_attrs(api, cfg->attrs, cfg->num_attrs, &flt);
  if (r) return r;
  if (!flt.attrs) {
    r = real(cfg, f, params, extra);
    if (r == WG_SUCCESS) count_after(ptsz, f, cfg->grid_x, cfg->grid_y, cfg->grid_z, cfg->stream);
    return r;
  }
  wg_cu_launch_config c = *cfg;
  c.attrs = flt.attrs;
  c.num_attrs = flt.n;
  r = real(&c, f, params, extra);
  free_filtered(&flt);
  if (r == WG_SUCCESS && flt.has_event) {
    event_record_fn rec = (event_record_fn)real_cuda(ptsz ? &r_event_record_ptsz : &r_event_record);
    if (!rec) return missing("cuEventRecordWithFlags");
    r = rec(flt.event, cfg->stream, (unsigned)flt.event_flags);
  }
  if (r == WG_SUCCESS) count_after(ptsz, f, cfg->grid_x, cfg->grid_y, cfg->grid_z, cfg->stream);
  return r;
}

WG_EXPORT wg_result cuLaunchKernelEx(const wg_cu_launch_config* cfg, void* f, void** params,
                                     void** extra) {
  return launch_ex(0, cfg, f, params, extra);
}

WG_EXPORT wg_result cuLaunchKernelEx_ptsz(const wg_cu_launch_config* cfg, void* f, void** params,
                                          void** extra) {
  return launch_ex(1, cfg, f, params, extra);
}

/* ---- function attributes and occupancy --------------------------------------------------- */
static int is_cluster_func_attr(int attrib) {
  return attrib >= WG_FUNC_CLUSTER_SIZE_MUST_BE_SET &&
         attrib <= WG_FUNC_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
}

/* Setting a cluster attribute: required dims 0 or 1 and the policy / portability switches are
 * the cluster-size-1 behaviour already; required dims above 1 cannot run. */
static wg_result set_cluster_attr(const char* api, int attrib, int value) {
  if (attrib >= WG_FUNC_REQUIRED_CLUSTER_WIDTH && attrib <= WG_FUNC_REQUIRED_CLUSTER_DEPTH) {
    if (value < 0) return fail(WG_ERROR_INVALID_VALUE, api, "attribute %d = %d", attrib, value);
    if (value > 1) {
      return fail(WG_ERROR_INVALID_CLUSTER_SIZE, api,
                  "required cluster dimension %d: the SM80 emulation runs cluster size 1 only",
                  value);
    }
  }
  notice_once(16u + (unsigned)attrib,
              "function attribute %d set to %d: no effect at cluster size 1", attrib, value);
  return WG_SUCCESS;
}

typedef wg_result (*func_set_fn)(void*, int, int);
typedef wg_result (*func_get_fn)(int*, int, void*);
typedef wg_result (*kernel_set_fn)(int, int, void*, int);
typedef wg_result (*kernel_get_fn)(int*, int, void*, int);

static real_sym r_func_set = {"cuFuncSetAttribute", NULL};
static real_sym r_func_get = {"cuFuncGetAttribute", NULL};
static real_sym r_kernel_set = {"cuKernelSetAttribute", NULL};
static real_sym r_kernel_get = {"cuKernelGetAttribute", NULL};

WG_EXPORT wg_result cuFuncSetAttribute(void* hfunc, int attrib, int value) {
  if (is_cluster_func_attr(attrib)) return set_cluster_attr("cuFuncSetAttribute", attrib, value);
  func_set_fn real = (func_set_fn)real_cuda(&r_func_set);
  return real ? real(hfunc, attrib, value) : missing("cuFuncSetAttribute");
}

/* Cluster attributes read as their defaults (0) [inferred: nvcc_emu removes compile-time
 * cluster dimensions, and the setters above store nothing]. */
WG_EXPORT wg_result cuFuncGetAttribute(int* pi, int attrib, void* hfunc) {
  if (is_cluster_func_attr(attrib)) {
    if (!pi) return fail(WG_ERROR_INVALID_VALUE, "cuFuncGetAttribute", "pi is NULL");
    *pi = 0;
    return WG_SUCCESS;
  }
  func_get_fn real = (func_get_fn)real_cuda(&r_func_get);
  return real ? real(pi, attrib, hfunc) : missing("cuFuncGetAttribute");
}

WG_EXPORT wg_result cuKernelSetAttribute(int attrib, int value, void* kernel, int dev) {
  if (is_cluster_func_attr(attrib)) return set_cluster_attr("cuKernelSetAttribute", attrib, value);
  kernel_set_fn real = (kernel_set_fn)real_cuda(&r_kernel_set);
  return real ? real(attrib, value, kernel, dev) : missing("cuKernelSetAttribute");
}

WG_EXPORT wg_result cuKernelGetAttribute(int* pi, int attrib, void* kernel, int dev) {
  if (is_cluster_func_attr(attrib)) {
    if (!pi) return fail(WG_ERROR_INVALID_VALUE, "cuKernelGetAttribute", "pi is NULL");
    *pi = 0;
    return WG_SUCCESS;
  }
  kernel_get_fn real = (kernel_get_fn)real_cuda(&r_kernel_get);
  return real ? real(pi, attrib, kernel, dev) : missing("cuKernelGetAttribute");
}

typedef wg_result (*occ_blocks_fn)(int*, void*, int, size_t, unsigned);
typedef wg_result (*ctx_get_device_fn)(int*);
typedef wg_result (*dev_attr_fn)(int*, int, int);

static real_sym r_occ_blocks = {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", NULL};
static real_sym r_ctx_get_device = {"cuCtxGetDevice", NULL};
static real_sym r_dev_attr = {"cuDeviceGetAttribute", NULL};
static real_sym r_dev_cc = {"cuDeviceComputeCapability", NULL};

/* Cluster dimension of a launch config: 1 when absent; error when it is not 1x1x1. */
static wg_result config_cluster_is_one(const char* api, const wg_cu_launch_config* cfg) {
  for (unsigned i = 0; cfg->attrs && i < cfg->num_attrs; ++i) {
    if (cfg->attrs[i].id != WG_ATTR_CLUSTER_DIMENSION) continue;
    const unsigned x = cfg->attrs[i].value.cluster_dim.x, y = cfg->attrs[i].value.cluster_dim.y,
                   z = cfg->attrs[i].value.cluster_dim.z;
    if (x != 1 || y != 1 || z != 1) {
      return fail(WG_ERROR_INVALID_CLUSTER_SIZE, api,
                  "cluster dimension %ux%ux%u: the SM80 emulation runs cluster size 1 only", x, y,
                  z);
    }
  }
  return WG_SUCCESS;
}

static wg_result blocks_per_sm(const char* api, void* func, const wg_cu_launch_config* cfg,
                               int* blocks) {
  occ_blocks_fn occ = (occ_blocks_fn)real_cuda(&r_occ_blocks);
  if (!occ) return missing(api);
  const int threads = (int)(cfg->block_x * cfg->block_y * cfg->block_z);
  return occ(blocks, func, threads, (size_t)cfg->shared_mem_bytes, 0u);
}

/* Clusters of one CTA that can co-exist: resident blocks per SM times the SM count. */
WG_EXPORT wg_result cuOccupancyMaxActiveClusters(int* num_clusters, void* func,
                                                 const wg_cu_launch_config* cfg) {
  static const char api[] = "cuOccupancyMaxActiveClusters";
  if (!num_clusters || !cfg) return fail(WG_ERROR_INVALID_VALUE, api, "NULL argument");
  wg_result r = config_cluster_is_one(api, cfg);
  if (r) return r;
  int blocks = 0, dev = 0, sms = 0;
  r = blocks_per_sm(api, func, cfg, &blocks);
  if (r) return r;
  ctx_get_device_fn get_dev = (ctx_get_device_fn)real_cuda(&r_ctx_get_device);
  dev_attr_fn attr = (dev_attr_fn)real_cuda(&r_dev_attr);
  if (!get_dev || !attr) return missing(api);
  r = get_dev(&dev);
  if (r) return r;
  r = attr(&sms, WG_DEV_MULTIPROCESSOR_COUNT, dev);
  if (r) return r;
  *num_clusters = blocks * sms;
  return WG_SUCCESS;
}

/* Largest cluster that fits: 1 when one CTA fits on an SM, else 0 [inferred]. */
WG_EXPORT wg_result cuOccupancyMaxPotentialClusterSize(int* cluster_size, void* func,
                                                       const wg_cu_launch_config* cfg) {
  static const char api[] = "cuOccupancyMaxPotentialClusterSize";
  if (!cluster_size || !cfg) return fail(WG_ERROR_INVALID_VALUE, api, "NULL argument");
  int blocks = 0;
  wg_result r = blocks_per_sm(api, func, cfg, &blocks);
  if (r) return r;
  *cluster_size = blocks > 0 ? 1 : 0;
  return WG_SUCCESS;
}

/* ---- compute capability spoofing ---------------------------------------------------------- */
static int g_spoof = -1;

static int spoof_cc(void) {
  int v = __atomic_load_n(&g_spoof, __ATOMIC_RELAXED);
  if (v < 0) {
    const char* e = getenv("WGEMU_SPOOF_CC");
    v = (e && strcmp(e, "1") == 0) ? 1 : 0;
    __atomic_store_n(&g_spoof, v, __ATOMIC_RELAXED);
  }
  if (v) notice_once(40u, "WGEMU_SPOOF_CC=1: reporting compute capability 9.0");
  return v;
}

WG_EXPORT wg_result cuDeviceGetAttribute(int* pi, int attrib, int dev) {
  dev_attr_fn real = (dev_attr_fn)real_cuda(&r_dev_attr);
  if (!real) return missing("cuDeviceGetAttribute");
  wg_result r = real(pi, attrib, dev);
  if (r != WG_SUCCESS || !pi) return r;
  switch (attrib) {
    case WG_DEV_COMPUTE_CAPABILITY_MAJOR:
      if (spoof_cc()) *pi = 9;
      break;
    case WG_DEV_COMPUTE_CAPABILITY_MINOR:
      if (spoof_cc()) *pi = 0;
      break;
    case WG_DEV_CLUSTER_LAUNCH:
    case WG_DEV_TENSOR_MAP_ACCESS_SUPPORTED:
      if (spoof_cc()) *pi = 1;
      break;
    default:
      break;
  }
  return r;
}

WG_EXPORT wg_result cuDeviceComputeCapability(int* major, int* minor, int dev) {
  typedef wg_result (*cc_fn)(int*, int*, int);
  cc_fn real = (cc_fn)real_cuda(&r_dev_cc);
  if (!real) return missing("cuDeviceComputeCapability");
  wg_result r = real(major, minor, dev);
  if (r == WG_SUCCESS && major && minor && spoof_cc()) {
    *major = 9;
    *minor = 0;
  }
  return r;
}

/* ---- entry-point substitution ------------------------------------------------------------ */
WG_EXPORT wg_result cuGetProcAddress_v2(const char* symbol, void** pfn, int cuda_version,
                                        uint64_t flags, int* status);
WG_EXPORT wg_result cuGetProcAddress(const char* symbol, void** pfn, int cuda_version,
                                     uint64_t flags);

static real_sym r_encode_tiled = {"cuTensorMapEncodeTiled", NULL};
static real_sym r_encode_im2col = {"cuTensorMapEncodeIm2col", NULL};
static real_sym r_replace_address = {"cuTensorMapReplaceAddress", NULL};
static real_sym r_occ_clusters = {"cuOccupancyMaxActiveClusters", NULL};
static real_sym r_occ_cluster_size = {"cuOccupancyMaxPotentialClusterSize", NULL};
static real_sym r_gpa = {"cuGetProcAddress", NULL};
static real_sym r_gpa_v2 = {"cuGetProcAddress_v2", NULL};

typedef struct wg_export {
  const char* name; /* exported symbol */
  const char* base; /* name passed to cuGetProcAddress / cudaGetDriverEntryPoint */
  int variant;      /* 1: _ptsz (per-thread stream) or _v2 (CUDA >= 12.0 ABI) */
  void* wrapper;
  real_sym* real;
} wg_export;

#define WG_FN(f) ((void*)(uintptr_t)(f))

static const wg_export k_exports[] = {
    {"cuTensorMapEncodeTiled", "cuTensorMapEncodeTiled", 0, WG_FN(cuTensorMapEncodeTiled),
     &r_encode_tiled},
    {"cuTensorMapEncodeIm2col", "cuTensorMapEncodeIm2col", 0, WG_FN(cuTensorMapEncodeIm2col),
     &r_encode_im2col},
    {"cuTensorMapReplaceAddress", "cuTensorMapReplaceAddress", 0, WG_FN(cuTensorMapReplaceAddress),
     &r_replace_address},
    {"cuLaunchKernel", "cuLaunchKernel", 0, WG_FN(cuLaunchKernel), &r_launch},
    {"cuLaunchKernel_ptsz", "cuLaunchKernel", 1, WG_FN(cuLaunchKernel_ptsz), &r_launch_ptsz},
    {"cuLaunchCooperativeKernel", "cuLaunchCooperativeKernel", 0, WG_FN(cuLaunchCooperativeKernel),
     &r_launch_coop},
    {"cuLaunchCooperativeKernel_ptsz", "cuLaunchCooperativeKernel", 1,
     WG_FN(cuLaunchCooperativeKernel_ptsz), &r_launch_coop_ptsz},
    {"cuLaunchKernelEx", "cuLaunchKernelEx", 0, WG_FN(cuLaunchKernelEx), &r_launch_ex},
    {"cuLaunchKernelEx_ptsz", "cuLaunchKernelEx", 1, WG_FN(cuLaunchKernelEx_ptsz),
     &r_launch_ex_ptsz},
    {"cuFuncSetAttribute", "cuFuncSetAttribute", 0, WG_FN(cuFuncSetAttribute), &r_func_set},
    {"cuFuncGetAttribute", "cuFuncGetAttribute", 0, WG_FN(cuFuncGetAttribute), &r_func_get},
    {"cuKernelSetAttribute", "cuKernelSetAttribute", 0, WG_FN(cuKernelSetAttribute), &r_kernel_set},
    {"cuKernelGetAttribute", "cuKernelGetAttribute", 0, WG_FN(cuKernelGetAttribute), &r_kernel_get},
    {"cuOccupancyMaxActiveClusters", "cuOccupancyMaxActiveClusters", 0,
     WG_FN(cuOccupancyMaxActiveClusters), &r_occ_clusters},
    {"cuOccupancyMaxPotentialClusterSize", "cuOccupancyMaxPotentialClusterSize", 0,
     WG_FN(cuOccupancyMaxPotentialClusterSize), &r_occ_cluster_size},
    {"cuDeviceGetAttribute", "cuDeviceGetAttribute", 0, WG_FN(cuDeviceGetAttribute), &r_dev_attr},
    {"cuDeviceComputeCapability", "cuDeviceComputeCapability", 0, WG_FN(cuDeviceComputeCapability),
     &r_dev_cc},
    {"cuGetProcAddress", "cuGetProcAddress", 0, WG_FN(cuGetProcAddress), &r_gpa},
    {"cuGetProcAddress_v2", "cuGetProcAddress", 1, WG_FN(cuGetProcAddress_v2), &r_gpa_v2},
};
#define WG_NUM_EXPORTS (sizeof(k_exports) / sizeof(k_exports[0]))

static const wg_export* find_export(const char* name) {
  if (name[0] != 'c' || name[1] != 'u') return NULL;
  for (size_t i = 0; i < WG_NUM_EXPORTS; ++i) {
    if (strcmp(k_exports[i].name, name) == 0) return &k_exports[i];
  }
  return NULL;
}

/* Replaces a driver entry point found for `base` by the shim's wrapper. The variant is matched
 * by address against the libcuda exports first; when the driver returned an address that is not
 * an export, the per-thread flag (launch) or the requested version (cuGetProcAddress, whose
 * 5-argument ABI started with CUDA 12.0, cuda.h line 159) decides [inferred]. */
static void* substitute_proc(const char* base, void* p, uint64_t flags, int cuda_version) {
  const wg_export* cand[2];
  int nc = 0;
  if (!base || !p) return p;
  for (size_t i = 0; i < WG_NUM_EXPORTS && nc < 2; ++i) {
    if (strcmp(k_exports[i].base, base) == 0) cand[nc++] = &k_exports[i];
  }
  if (nc == 0) return p;
  for (int i = 0; i < nc; ++i) {
    if (p == cand[i]->wrapper) return p;
  }
  for (int i = 0; i < nc; ++i) {
    if (p == real_cuda(cand[i]->real)) return cand[i]->wrapper;
  }
  if (nc == 1) return cand[0]->wrapper;
  const int want = strcmp(base, "cuGetProcAddress") == 0
                       ? (cuda_version >= 12000)
                       : ((flags & WG_PROC_PER_THREAD_DEFAULT_STREAM) != 0);
  return cand[0]->variant == want ? cand[0]->wrapper : cand[1]->wrapper;
}

typedef wg_result (*gpa_v2_fn)(const char*, void**, int, uint64_t, int*);
typedef wg_result (*gpa_fn)(const char*, void**, int, uint64_t);

WG_EXPORT wg_result cuGetProcAddress_v2(const char* symbol, void** pfn, int cuda_version,
                                        uint64_t flags, int* status) {
  gpa_v2_fn real = (gpa_v2_fn)real_cuda(&r_gpa_v2);
  if (!real) return missing("cuGetProcAddress_v2");
  wg_result r = real(symbol, pfn, cuda_version, flags, status);
  if (r == WG_SUCCESS && pfn && (!status || *status == 0)) {
    *pfn = substitute_proc(symbol, *pfn, flags, cuda_version);
  }
  return r;
}

WG_EXPORT wg_result cuGetProcAddress(const char* symbol, void** pfn, int cuda_version,
                                     uint64_t flags) {
  gpa_fn real = (gpa_fn)real_cuda(&r_gpa);
  if (!real) return missing("cuGetProcAddress");
  wg_result r = real(symbol, pfn, cuda_version, flags);
  if (r == WG_SUCCESS && pfn) *pfn = substitute_proc(symbol, *pfn, flags, cuda_version);
  return r;
}

/* Intercepted names only: the address is replaced when it is libcuda's export. The lookup runs
 * from the shim, so RTLD_NEXT means "after the shim" for these names [inferred: the shim is first
 * in LD_PRELOAD, which puts libcuda after it]. */
static void* dlsym_intercepted(dlsym_fn real, void* handle, const char* name, const wg_export* e) {
  void* p = real(handle, name);
  if (!p || p == e->wrapper || !cuda_handle(0)) return p;
  return p == real_cuda(e->real) ? e->wrapper : p;
}

/* Every other name is a tail call, so glibc sees the caller's return address and RTLD_NEXT keeps
 * its meaning (tests/async/host/check_shim_tailcall.sh checks the jump). */
WG_EXPORT void* dlsym(void* handle, const char* name) {
  dlsym_fn real = real_dlsym();
  const wg_export* e = name ? find_export(name) : NULL;
  if (!e) return real(handle, name);
  return dlsym_intercepted(real, handle, name, e);
}

/* ---- CUDA runtime ------------------------------------------------------------------------ */
typedef wg_result (*rt_entry_fn)(const char*, void**, unsigned long long, int*);
typedef wg_result (*rt_entry_ver_fn)(const char*, void**, unsigned, unsigned long long, int*);
typedef wg_result (*rt_launch_ex_fn)(const wg_rt_launch_config*, const void*, void**);
typedef wg_result (*rt_event_record_fn)(void*, void*, unsigned);

static real_sym r_rt_entry = {"cudaGetDriverEntryPoint", NULL};
static real_sym r_rt_entry_ptsz = {"cudaGetDriverEntryPoint_ptsz", NULL};
static real_sym r_rt_entry_ver = {"cudaGetDriverEntryPointByVersion", NULL};
static real_sym r_rt_entry_ver_ptsz = {"cudaGetDriverEntryPointByVersion_ptsz", NULL};
static real_sym r_rt_launch_ex = {"cudaLaunchKernelExC", NULL};
static real_sym r_rt_launch_ex_ptsz = {"cudaLaunchKernelExC_ptsz", NULL};
static real_sym r_rt_event_record = {"cudaEventRecordWithFlags", NULL};
static real_sym r_rt_event_record_ptsz = {"cudaEventRecordWithFlags_ptsz", NULL};

/* cudaEnableDefault through a _ptsz entry means the per-thread default stream. */
static uint64_t rt_flags(unsigned long long flags, int ptsz) {
  uint64_t f = (uint64_t)flags;
  if (ptsz && !(f & (WG_PROC_LEGACY_STREAM | WG_PROC_PER_THREAD_DEFAULT_STREAM))) {
    f |= WG_PROC_PER_THREAD_DEFAULT_STREAM;
  }
  return f;
}

static wg_result rt_entry(int ptsz, const char* symbol, void** fp, unsigned long long flags,
                          int* status) {
  rt_entry_fn real = (rt_entry_fn)real_cudart(ptsz ? &r_rt_entry_ptsz : &r_rt_entry);
  if (!real) return missing("cudaGetDriverEntryPoint");
  wg_result r = real(symbol, fp, flags, status);
  /* No version argument: this runtime is 12.0 or newer, so the 5-argument cuGetProcAddress. */
  if (r == WG_SUCCESS && fp && (!status || *status == 0)) {
    *fp = substitute_proc(symbol, *fp, rt_flags(flags, ptsz), 12000);
  }
  return r;
}

static wg_result rt_entry_ver(int ptsz, const char* symbol, void** fp, unsigned version,
                              unsigned long long flags, int* status) {
  rt_entry_ver_fn real =
      (rt_entry_ver_fn)real_cudart(ptsz ? &r_rt_entry_ver_ptsz : &r_rt_entry_ver);
  if (!real) return missing("cudaGetDriverEntryPointByVersion");
  wg_result r = real(symbol, fp, version, flags, status);
  if (r == WG_SUCCESS && fp && (!status || *status == 0)) {
    *fp = substitute_proc(symbol, *fp, rt_flags(flags, ptsz), (int)version);
  }
  return r;
}

WG_EXPORT wg_result cudaGetDriverEntryPoint(const char* symbol, void** fp, unsigned long long flags,
                                            int* status) {
  return rt_entry(0, symbol, fp, flags, status);
}

WG_EXPORT wg_result cudaGetDriverEntryPoint_ptsz(const char* symbol, void** fp,
                                                 unsigned long long flags, int* status) {
  return rt_entry(1, symbol, fp, flags, status);
}

WG_EXPORT wg_result cudaGetDriverEntryPointByVersion(const char* symbol, void** fp,
                                                     unsigned version, unsigned long long flags,
                                                     int* status) {
  return rt_entry_ver(0, symbol, fp, version, flags, status);
}

WG_EXPORT wg_result cudaGetDriverEntryPointByVersion_ptsz(const char* symbol, void** fp,
                                                          unsigned version,
                                                          unsigned long long flags, int* status) {
  return rt_entry_ver(1, symbol, fp, version, flags, status);
}

static wg_result rt_launch_ex(int ptsz, const wg_rt_launch_config* cfg, const void* func,
                              void** args) {
  const char* api = ptsz ? "cudaLaunchKernelExC_ptsz" : "cudaLaunchKernelExC";
  rt_launch_ex_fn real =
      (rt_launch_ex_fn)real_cudart(ptsz ? &r_rt_launch_ex_ptsz : &r_rt_launch_ex);
  if (!real) return missing(api);
  if (!cfg) return real(cfg, func, args);
  wg_filtered flt;
  wg_result r = filter_attrs(api, cfg->attrs, cfg->num_attrs, &flt);
  if (r) return r;
  if (!flt.attrs) return real(cfg, func, args);
  wg_rt_launch_config c = *cfg;
  c.attrs = flt.attrs;
  c.num_attrs = flt.n;
  r = real(&c, func, args);
  free_filtered(&flt);
  if (r == WG_SUCCESS && flt.has_event) {
    rt_event_record_fn rec =
        (rt_event_record_fn)real_cudart(ptsz ? &r_rt_event_record_ptsz : &r_rt_event_record);
    if (!rec) return missing("cudaEventRecordWithFlags");
    r = rec(flt.event, cfg->stream, (unsigned)flt.event_flags);
  }
  return r;
}

WG_EXPORT wg_result cudaLaunchKernelExC(const wg_rt_launch_config* cfg, const void* func,
                                        void** args) {
  return rt_launch_ex(0, cfg, func, args);
}

WG_EXPORT wg_result cudaLaunchKernelExC_ptsz(const wg_rt_launch_config* cfg, const void* func,
                                             void** args) {
  return rt_launch_ex(1, cfg, func, args);
}
