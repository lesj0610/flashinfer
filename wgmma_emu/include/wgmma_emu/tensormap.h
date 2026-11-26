/* Emulated tensor-map format for the SM80 wgmma emulation layer (fork-only). Owner: A.
 *
 * The 128-byte CUtensorMap object is opaque (ptx.txt 4039-4043). On SM80 the driver refuses to
 * encode one (cuTensorMapEncodeTiled returns 801), so host/shim.c writes this format instead, and
 * the device library (runtime/lib_async.cu) reads it for cp.async.bulk.tensor and rewrites it for
 * tensormap.replace. Host and device share this header; it is plain C (C99) and C++ compatible.
 *
 * Layout (little endian, 128 bytes, see wgemu_tmap below):
 *   [  0,   8) global_address           tensor base address (generic == global address)
 *   [  8,  12) misc                     bit fields WGEMU_TM_* below
 *   [ 12,  16) magic                    WGEMU_TMAP_MAGIC
 *   [ 16,  36) global_dim_m1[5]         size of each dimension minus one (sizes up to 2^32)
 *   [ 36,  40) reserved0                zero
 *   [ 40,  72) global_stride[4]         byte stride of dimensions 1..4 (dimension 0 is dense)
 *   [ 72,  77) box_dim_m1[5]            tile mode: bounding box size minus one (box <= 256)
 *   [ 77,  80) reserved1                zero
 *   [ 80,  85) element_stride_m1[5]     traversal stride minus one (stride 1..8)
 *   [ 85,  86) channels_m1              im2col: channelsPerPixel minus one (<= 256)
 *   [ 86,  88) pixels_m1                im2col: pixelsPerColumn minus one (<= 1024)
 *   [ 88,  94) lower_corner[3]          im2col: bounding-box lower corner, index 0 = W (dim 1)
 *   [ 94, 100) upper_corner[3]          im2col: bounding-box upper corner, index 0 = W (dim 1)
 *   [100, 128) reserved2                zero
 *
 * misc bit 21 is the tensor allocation class (1 = large). CuTe clears bit 21 of qword 1 after
 * encoding small tensors on drivers <= 13010 (cute/atom/copy_traits_sm90_tma.hpp:1064-1068); qword
 * 1 is bytes [8, 16), so its bit 21 is misc bit 21 and that store keeps meaning "small".
 *
 * Element types use the PTX tensormap.replace .elemtype codes (ptx.txt Table 36), which differ from
 * CUtensorMapDataType; the shim converts. Swizzle uses the PTX .swizzle_mode codes (0 none, 1 32B,
 * 2 64B, 3 128B, 4 96B) plus .swizzle_atomicity (0 16B, 1 32B, 2 32B+8B flip, 3 64B).
 */
#ifndef WGEMU_TENSORMAP_H
#define WGEMU_TENSORMAP_H

#include <stdint.h>

#if defined(__CUDACC__)
#define WGEMU_TMAP_FN __host__ __device__ static inline
#else
#define WGEMU_TMAP_FN static inline
#endif

#define WGEMU_TMAP_MAGIC 0x31544D57u /* "WMT1" */
#define WGEMU_TMAP_BYTES 128

/* PTX .elemtype codes (ptx.txt Table 36). */
#define WGEMU_TM_U8 0
#define WGEMU_TM_U16 1
#define WGEMU_TM_U32 2
#define WGEMU_TM_S32 3
#define WGEMU_TM_U64 4
#define WGEMU_TM_S64 5
#define WGEMU_TM_F16 6
#define WGEMU_TM_F32 7
#define WGEMU_TM_F32_FTZ 8
#define WGEMU_TM_F64 9
#define WGEMU_TM_BF16 10
#define WGEMU_TM_TF32 11
#define WGEMU_TM_TF32_FTZ 12
#define WGEMU_TM_B4X16 13
#define WGEMU_TM_B4X16_P64 14
#define WGEMU_TM_B6X16_P32 15

/* PTX .interleave_layout, .swizzle_mode, .swizzle_atomicity, .fill_mode codes. */
#define WGEMU_TM_INTERLEAVE_NONE 0
#define WGEMU_TM_INTERLEAVE_16B 1
#define WGEMU_TM_INTERLEAVE_32B 2
#define WGEMU_TM_SWIZZLE_NONE 0
#define WGEMU_TM_SWIZZLE_32B 1
#define WGEMU_TM_SWIZZLE_64B 2
#define WGEMU_TM_SWIZZLE_128B 3
#define WGEMU_TM_SWIZZLE_96B 4
#define WGEMU_TM_ATOM_16B 0
#define WGEMU_TM_ATOM_32B 1
#define WGEMU_TM_ATOM_32B_FLIP_8B 2
#define WGEMU_TM_ATOM_64B 3
#define WGEMU_TM_FILL_ZERO 0
#define WGEMU_TM_FILL_NAN 1

/* misc bit fields: (shift, width). */
#define WGEMU_TM_RANK_M1_SHIFT 0      /* 3 bits: rank - 1 */
#define WGEMU_TM_ELEMTYPE_SHIFT 3     /* 4 bits */
#define WGEMU_TM_INTERLEAVE_SHIFT 7   /* 2 bits */
#define WGEMU_TM_SWIZZLE_SHIFT 9      /* 3 bits */
#define WGEMU_TM_ATOMICITY_SHIFT 12   /* 2 bits */
#define WGEMU_TM_FILL_SHIFT 14        /* 1 bit */
#define WGEMU_TM_IM2COL_SHIFT 15      /* 1 bit: 0 tile, 1 im2col */
#define WGEMU_TM_L2PROMO_SHIFT 16     /* 2 bits: CUtensorMapL2promotion, informational */
#define WGEMU_TM_ALLOC_LARGE_SHIFT 21 /* 1 bit: tensor allocation class large */

typedef struct wgemu_tmap {
  uint64_t global_address;
  uint32_t misc;
  uint32_t magic;
  uint32_t global_dim_m1[5];
  uint32_t reserved0;
  uint64_t global_stride[4];
  uint8_t box_dim_m1[5];
  uint8_t reserved1[3];
  uint8_t element_stride_m1[5];
  uint8_t channels_m1;
  uint16_t pixels_m1;
  int16_t lower_corner[3];
  int16_t upper_corner[3];
  uint8_t reserved2[28];
} wgemu_tmap;

#if defined(__cplusplus)
#define WGEMU_TMAP_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define WGEMU_TMAP_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
WGEMU_TMAP_STATIC_ASSERT(sizeof(wgemu_tmap) == WGEMU_TMAP_BYTES, "wgemu_tmap must be 128 bytes");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, global_stride) == 40, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, box_dim_m1) == 72, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, element_stride_m1) == 80, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, pixels_m1) == 86, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, lower_corner) == 88, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, upper_corner) == 94, "layout");
WGEMU_TMAP_STATIC_ASSERT(__builtin_offsetof(wgemu_tmap, reserved2) == 100, "layout");

WGEMU_TMAP_FN uint32_t wgemu_tm_get(uint32_t misc, int shift, int width) {
  return (misc >> shift) & ((1u << width) - 1u);
}

WGEMU_TMAP_FN uint32_t wgemu_tm_set(uint32_t misc, int shift, int width, uint32_t v) {
  uint32_t mask = ((1u << width) - 1u) << shift;
  return (misc & ~mask) | ((v << shift) & mask);
}

WGEMU_TMAP_FN int wgemu_tm_rank(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_RANK_M1_SHIFT, 3) + 1;
}
WGEMU_TMAP_FN int wgemu_tm_elemtype(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_ELEMTYPE_SHIFT, 4);
}
WGEMU_TMAP_FN int wgemu_tm_interleave(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_INTERLEAVE_SHIFT, 2);
}
WGEMU_TMAP_FN int wgemu_tm_swizzle(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_SWIZZLE_SHIFT, 3);
}
WGEMU_TMAP_FN int wgemu_tm_atomicity(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_ATOMICITY_SHIFT, 2);
}
WGEMU_TMAP_FN int wgemu_tm_fill_nan(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_FILL_SHIFT, 1);
}
WGEMU_TMAP_FN int wgemu_tm_is_im2col(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_IM2COL_SHIFT, 1);
}
WGEMU_TMAP_FN int wgemu_tm_alloc_large(const wgemu_tmap* t) {
  return (int)wgemu_tm_get(t->misc, WGEMU_TM_ALLOC_LARGE_SHIFT, 1);
}

/* Element size in bytes for a PTX .elemtype code; 0 for the packed sub-byte types (sm_100+). */
WGEMU_TMAP_FN int wgemu_tm_elem_bytes(int elemtype) {
  switch (elemtype) {
    case WGEMU_TM_U8:
      return 1;
    case WGEMU_TM_U16:
    case WGEMU_TM_F16:
    case WGEMU_TM_BF16:
      return 2;
    case WGEMU_TM_U32:
    case WGEMU_TM_S32:
    case WGEMU_TM_F32:
    case WGEMU_TM_F32_FTZ:
    case WGEMU_TM_TF32:
    case WGEMU_TM_TF32_FTZ:
      return 4;
    case WGEMU_TM_U64:
    case WGEMU_TM_S64:
    case WGEMU_TM_F64:
      return 8;
    default:
      return 0;
  }
}

WGEMU_TMAP_FN int wgemu_tm_elem_is_float(int elemtype) {
  return elemtype == WGEMU_TM_F16 || elemtype == WGEMU_TM_F32 || elemtype == WGEMU_TM_F32_FTZ ||
         elemtype == WGEMU_TM_F64 || elemtype == WGEMU_TM_BF16 || elemtype == WGEMU_TM_TF32 ||
         elemtype == WGEMU_TM_TF32_FTZ;
}

/* Swizzle span in bytes for a PTX .swizzle_mode code (0 for none). */
WGEMU_TMAP_FN int wgemu_tm_swizzle_bytes(int mode) {
  switch (mode) {
    case WGEMU_TM_SWIZZLE_32B:
      return 32;
    case WGEMU_TM_SWIZZLE_64B:
      return 64;
    case WGEMU_TM_SWIZZLE_128B:
      return 128;
    case WGEMU_TM_SWIZZLE_96B:
      return 96;
    default:
      return 0;
  }
}

/* Swizzle<B,4,3> on an absolute shared-memory byte address (ptx.txt 3552-4038 tables: the 16-byte
 * chunk index inside each 128-byte line is XORed with the line index modulo 2/4/8 for 32/64/128B).
 * Only modes 0..3 are valid here; callers reject 96B and non-16B atomicity first. */
WGEMU_TMAP_FN uint32_t wgemu_tm_swizzle_addr(uint32_t addr, int mode) {
  uint32_t bbits = (mode == WGEMU_TM_SWIZZLE_32B)    ? 1u
                   : (mode == WGEMU_TM_SWIZZLE_64B)  ? 2u
                   : (mode == WGEMU_TM_SWIZZLE_128B) ? 3u
                                                     : 0u;
  uint32_t mask = (1u << bbits) - 1u;
  return addr ^ (((addr >> 7) & mask) << 4);
}

#endif /* WGEMU_TENSORMAP_H */
