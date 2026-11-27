// SM90 wgmma layout facts for the SM80 emulation layer (fork-only). Owner: S.
//
// Everything here is pure index arithmetic: register-fragment coordinates,
// sparse metadata placement, matrix-descriptor fields and shared-memory
// addresses of the canonical GMMA layouts. No numerics, no memory access.
//
// Entry points (PLAN.md section 12 signatures, namespace wgemu::spec):
//   int      k_of(EType atype, bool sparse);
//   bool     valid_n(EType atype, int n);
//   void     d_frag_coord(int t, int i, int* m, int* n);
//   void     a_frag_coord(EType at, int t, int j, int e, int* m, int* k);
//   void     sp_a_frag_coord(EType at, int t, int j, int e, int* m, int* kc);
//   void     sp_meta_loc(EType at, int sel, int m, int chunk, int* lane, int* bit);
//   void     sp_meta_decode(EType at, uint32_t field, int* idx0, int* idx1);
//   struct   Desc { uint32_t start, lbo, sbo, base_off; int swizzle_bytes; };
//   Desc     decode_desc(uint64_t d);
//   uint64_t encode_desc(const Desc& d);
//   uint32_t smem_addr(const Desc& d, bool mn_major, int elem_bits, int mn, int k);
//   uint32_t swizzle_addr(uint32_t addr, int swizzle_bytes, uint32_t base_off);
// Extra helpers (not in section 12, safe to use):
//   int  a_elems_per_reg(EType at);            // 32 / element bits (b1: 32)
//   int  sp_chunk_width(EType at);             // logical chunk width: 4 (2:4) or 2 (tf32 1:2)
//   int  sp_nnz_per_chunk(EType at);           // stored elements per chunk: 2 or 1
//   int  sp_logical_k(EType at, int kc, int idx);  // logical column of compressed kc, index idx
//   bool sp_meta_valid(EType at, uint32_t field);  // false for values PTX calls undefined
//   int  sp_meta_chunks(EType at);             // chunks per row of one sparse wgmma: 8 or 16
//   int  swizzle_code(int swizzle_bytes);      // bits 63-62 for 0/32/64/128 bytes, else -1
//   bool trans_allowed(EType at);              // imm-trans-a/b exists (f16/bf16 SS only)
//
// Conventions:
//   t = thread index in the warpgroup (tid % 128), w = t / 32, lane l = t % 32,
//   g = l / 4, q = l % 4, r = 16 w + g.
//   Desc fields hold decoded byte values: start, lbo and sbo are the 14-bit
//   descriptor fields shifted left by 4; base_off is the raw 3-bit field;
//   swizzle_bytes is 0 (none), 32, 64 or 128.
//   b1 operands are handled as 32-bit words by smem_addr (k = word index,
//   eight words per 256-bit row) and as single bits by a_frag_coord (e = bit).
//
// Sources: PTX ISA 9.5 section 9.7.17 (ext/ptxisa/wgmma_spec.txt and the
// figures in ext/ptxisa/img), cross-checked against CuTe
// (cute/atom/mma_traits_sm90_gmma*.hpp) by tests/spec/test_spec_cpu.cu.
// Points that the sources leave open are tagged [inferred] below.
#pragma once

#include "types.h"

namespace wgemu {
namespace spec {

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

// Logical K of one wgmma. Dense: f16/bf16 16, tf32 8, 8-bit 32, b1 256.
// Sparse (logical, before compression): f16/bf16 32, tf32 16, 8-bit 64.
// Returns 0 for combinations that do not exist (sparse b1, f32/s32 inputs).
WGEMU_HD int k_of(EType atype, bool sparse) {
  switch (atype) {
    case EType::F16:
    case EType::BF16:
      return sparse ? 32 : 16;
    case EType::TF32:
      return sparse ? 16 : 8;
    case EType::E4M3:
    case EType::E5M2:
    case EType::S8:
    case EType::U8:
      return sparse ? 64 : 32;
    case EType::B1:
      return sparse ? 0 : 256;
    default:
      return 0;
  }
}

// Valid N for the input type (same list for dense and sparse).
// Floating point inputs: 8..256 step 8.
// Integer and b1 inputs: 8, 16, 24, 32, then 48..256 step 16. The dense
// integer syntax block of PTX 9.5 stops at n224 (wgmma_spec.txt:952-955) but
// the shape table (wgmma_spec.txt:96-100), the D fragment text
// (wgmma_spec.txt:350-351) and the sparse integer syntax (wgmma_spec.txt:
// 1417-1421) include n240 and n256, so the superset is accepted.
WGEMU_HD bool valid_n(EType atype, int n) {
  if (n < 8 || n > 256 || (n % 8) != 0) return false;
  switch (atype) {
    case EType::F16:
    case EType::BF16:
    case EType::TF32:
    case EType::E4M3:
    case EType::E5M2:
      return true;
    case EType::S8:
    case EType::U8:
    case EType::B1:
      return n <= 32 || (n % 16) == 0;
    default:
      return false;
  }
}

// imm-trans-a / imm-trans-b exist only for f16/bf16 with operands read
// through descriptors (wgmma_spec.txt:984-988, 1484-1488).
WGEMU_HD bool trans_allowed(EType at) { return at == EType::F16 || at == EType::BF16; }

// ---------------------------------------------------------------------------
// Register fragments
// ---------------------------------------------------------------------------

// Accumulator D, element i in [0, N/2): same placement for every dtype and
// every K (figures 152, 154, 156, including the s32 D of k32 integer MMA;
// CuTe CLayout_64xN for every GMMA atom; wgmma_spec.txt:1209-1212 says the
// sparse D is the dense D of the same dtype). f32/s32 D: element i is
// register i. f16 D: register i holds elements 2i (low half) and 2i+1.
// Figure 158 (k256, s32 D for b1) draws a different pattern; this function
// keeps the uniform layout for b1 as well [inferred]. See tests/spec for the
// recorded conflict.
WGEMU_HD void d_frag_coord(int t, int i, int* m, int* n) {
  const int tt = t & 127;
  const int w = tt >> 5, l = tt & 31, g = l >> 2, q = l & 3;
  const int r = 16 * w + g;
  *m = r + 8 * ((i & 3) >> 1);
  *n = 8 * (i >> 2) + 2 * q + (i & 1);
}

// Elements per 32-bit A register: f16/bf16 2, tf32 1, 8-bit 4, b1 32.
WGEMU_HD int a_elems_per_reg(EType at) {
  const int b = etype_bits(at);
  return b > 0 ? 32 / b : 0;
}

// Dense A from registers: register j in 0..3, element e in the register
// (low to high; b1: bit e in 0..31). Figures 151 (k16), 153 (k8),
// 155 (k32), 157 (k256); CuTe ALayout_64x8/16/32.
WGEMU_HD void a_frag_coord(EType at, int t, int j, int e, int* m, int* k) {
  const int tt = t & 127;
  const int w = tt >> 5, l = tt & 31, g = l >> 2, q = l & 3;
  const int K = k_of(at, false);
  *m = 16 * w + g + 8 * (j & 1);
  *k = (j >> 1) * (K / 2) + q * (K / 8) + e;
}

// Sparse A from registers, compressed column kc in [0, K/2): the dense rule
// with K replaced by K/2. Figures 178 (f16/bf16 k32), 180 (tf32 k16),
// 182 (8-bit k64); CuTe sparse traits use ALayout_64x32/16/64 in logical K,
// where logical columns 2kc and 2kc+1 pack into compressed element kc.
WGEMU_HD void sp_a_frag_coord(EType at, int t, int j, int e, int* m, int* kc) {
  const int tt = t & 127;
  const int w = tt >> 5, l = tt & 31, g = l >> 2, q = l & 3;
  const int Kc = k_of(at, true) / 2;
  *m = 16 * w + g + 8 * (j & 1);
  *kc = (j >> 1) * (Kc / 2) + q * (Kc / 8) + e;
}

// ---------------------------------------------------------------------------
// Sparse metadata
// ---------------------------------------------------------------------------

// Logical width of one sparsity chunk and the number of stored elements in
// it: f16/bf16/8-bit 2:4, tf32 1:2 (wgmma_spec.txt:1142-1180).
WGEMU_HD int sp_chunk_width(EType at) { return at == EType::TF32 ? 2 : 4; }
WGEMU_HD int sp_nnz_per_chunk(EType at) { return at == EType::TF32 ? 1 : 2; }
// Chunks per row of one sparse wgmma: f16/bf16 32/4, tf32 16/2, 8-bit 64/4.
WGEMU_HD int sp_meta_chunks(EType at) { return k_of(at, true) / sp_chunk_width(at); }

// Logical column of compressed column kc when its metadata index is idx
// (idx from sp_meta_decode: idx0 for the lower stored element of a chunk,
// idx1 for the upper one; figures 174-177 put the first non-zero element in
// the low half of the operand and its index in metadata bits [1:0]).
WGEMU_HD int sp_logical_k(EType at, int kc, int idx) {
  return (kc / sp_nnz_per_chunk(at)) * sp_chunk_width(at) + idx;
}

// Location of the 4-bit metadata field for row m (0..63) and chunk `chunk`.
// The field lives in the sp-meta register of lane `lane` of warp m / 16
// (warpgroup thread 32 * (m / 16) + lane), bits [bit, bit + 3].
//   f16/bf16 (8 chunks) and tf32 (8 chunks): lane = 4g + 2 sel + (chunk >= 4),
//     bit = 16 hi + 4 (chunk % 4); sel selects the thread pair (T0,T1) or
//     (T2,T3) of each group of four (figures 179, 181).
//   8-bit (16 chunks): lane = 4g + hi + 2 (chunk >= 8), bit = 4 (chunk % 8);
//     all lanes contribute and sel must be 0 (figures 183, 184).
// with rr = m % 16, g = rr % 8, hi = rr / 8.
WGEMU_HD void sp_meta_loc(EType at, int sel, int m, int chunk, int* lane, int* bit) {
  const int rr = m & 15, g = rr & 7, hi = rr >> 3;
  if (etype_bits(at) == 8) {
    *lane = 4 * g + hi + 2 * (chunk >= 8 ? 1 : 0);
    *bit = 4 * (chunk & 7);
  } else {
    *lane = 4 * g + 2 * (sel & 1) + (chunk >= 4 ? 1 : 0);
    *bit = 16 * hi + 4 * (chunk & 3);
  }
}

// Decode one 4-bit metadata field.
//   2:4 types: idx0 = bits[1:0] (lower stored element), idx1 = bits[3:2].
//   tf32: the only meaningful values are 0b0100 (element 0) and 0b1110
//   (element 1) (wgmma_spec.txt:1155-1158, figure 175); idx0 = (field & 3) >> 1
//   and idx1 = -1.
WGEMU_HD void sp_meta_decode(EType at, uint32_t field, int* idx0, int* idx1) {
  if (at == EType::TF32) {
    *idx0 = static_cast<int>((field & 3u) >> 1);
    *idx1 = -1;
  } else {
    *idx0 = static_cast<int>(field & 3u);
    *idx1 = static_cast<int>((field >> 2) & 3u);
  }
}

// 2:4: 0b0000, 0b0101, 0b1010, 0b1111 are undefined (wgmma_spec.txt:1135-1136).
// tf32: only 0b0100 and 0b1110 are defined (wgmma_spec.txt:1157-1158).
WGEMU_HD bool sp_meta_valid(EType at, uint32_t field) {
  field &= 15u;
  if (at == EType::TF32) return field == 0x4u || field == 0xEu;
  return (field & 3u) != ((field >> 2) & 3u);
}

// ---------------------------------------------------------------------------
// Matrix descriptor (wgmma_spec.txt:826-893)
// ---------------------------------------------------------------------------

struct Desc {
  uint32_t start, lbo, sbo, base_off;
  int swizzle_bytes; /* 0, 32, 64, 128 */
};

// Bits 63-62: 0 none, 1 128B, 2 64B, 3 32B. Returns -1 for other sizes.
WGEMU_HD int swizzle_code(int swizzle_bytes) {
  switch (swizzle_bytes) {
    case 0:
      return 0;
    case 128:
      return 1;
    case 64:
      return 2;
    case 32:
      return 3;
    default:
      return -1;
  }
}

WGEMU_HD Desc decode_desc(uint64_t d) {
  Desc r;
  r.start = static_cast<uint32_t>(d & 0x3FFFu) << 4;
  r.lbo = static_cast<uint32_t>((d >> 16) & 0x3FFFu) << 4;
  r.sbo = static_cast<uint32_t>((d >> 32) & 0x3FFFu) << 4;
  r.base_off = static_cast<uint32_t>((d >> 49) & 0x7u);
  const int code = static_cast<int>((d >> 62) & 0x3u);
  r.swizzle_bytes = code == 0 ? 0 : (code == 1 ? 128 : (code == 2 ? 64 : 32));
  return r;
}

// matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4 for start, LBO and SBO.
// Precondition: swizzle_bytes is 0, 32, 64 or 128 (anything else encodes as
// code 0 = no swizzle; callers must not rely on that).
WGEMU_HD uint64_t encode_desc(const Desc& d) {
  const int code = swizzle_code(d.swizzle_bytes);
  uint64_t r = 0;
  r |= static_cast<uint64_t>((d.start & 0x3FFFFu) >> 4);
  r |= static_cast<uint64_t>((d.lbo & 0x3FFFFu) >> 4) << 16;
  r |= static_cast<uint64_t>((d.sbo & 0x3FFFFu) >> 4) << 32;
  r |= static_cast<uint64_t>(d.base_off & 0x7u) << 49;
  r |= static_cast<uint64_t>(code < 0 ? 0 : code) << 62;
  return r;
}

// Swizzle<B,4,3> on the byte address: bits [4, 4+B) ^= bits [7, 7+B) with
// B = 1 (32B), 2 (64B), 3 (128B). The row phase is taken relative to the
// pattern start: phase = ((addr >> 7) - base_off) & mask. With base_off = 0
// this is the absolute-address swizzle that CuTe uses (CuTe always encodes
// base_offset 0). For base_off != 0 the PTX text only defines how base_off is
// computed from the pattern start ((start >> 7) & 7, wgmma_spec.txt:874-893);
// that the hardware then swizzles relative to that start is [inferred]. PTX
// says base_off applies to every swizzle mode (wgmma_spec.txt:860); the CuTe
// bitfield comment says 64B/128B only (cute/arch/mma_sm90_desc.hpp:120-121);
// this follows PTX.
WGEMU_HD uint32_t swizzle_addr(uint32_t addr, int swizzle_bytes, uint32_t base_off) {
  int B;
  switch (swizzle_bytes) {
    case 32:
      B = 1;
      break;
    case 64:
      B = 2;
      break;
    case 128:
      B = 3;
      break;
    default:
      return addr;
  }
  const uint32_t mask = (1u << B) - 1u;
  const uint32_t phase = ((addr >> 7) - base_off) & mask;
  return addr ^ (phase << 4);
}

// Shared-memory byte address of logical element (mn, k) of the operand the
// descriptor describes (mn = row of A in M, or column of B in N; k = column
// along K, compressed for sparse A). elem_bits is 8, 16 or 32 (b1: 32-bit
// words, k = word index). Canonical layouts (wgmma_spec.txt:566-825, CuTe
// make_gmma_desc comment), with eb = elem_bits / 8 and W = swizzle bytes:
//   K-major, none : r0 = mn % 8, r1 = mn / 8, kb = k * eb
//                   r0 * 16 + r1 * SBO + kb % 16 + (kb / 16) * LBO
//   K-major, W    : r0 * W + r1 * SBO + kb        (LBO unused, "assumed 1",
//                   i.e. consecutive 16-byte columns are 16 bytes apart)
//   MN-major, none: mb = mn * eb
//                   mb % 16 + (mb / 16) * SBO + (k % 8) * 16 + (k / 8) * LBO
//   MN-major, W   : mb % W + (mb / W) * LBO + (k % 8) * W + (k / 8) * SBO
// then swizzle_addr(start + offset, W, base_off).
// Dense wgmma and sparse A read 32 bytes of K per row; sparse B reads 64
// bytes. For K-major swizzled layouts with kb >= W (only sparse B with 32B
// swizzle can need it) the linear continuation above is [inferred]: PTX gives
// no LBO for that case, and CuTe's tile_to_shape would place the second
// 32-byte column block elsewhere. Check mode reports that descriptor as a
// violation (sparse_b_layout_undefined); exact and fast keep the continuation
// and their results for it are not evidence (PLAN section 7).
// Sparse B (64 bytes of K per row) in K-major with 32B swizzle: PTX defines no
// address for bytes 32..63 of a row (wgmma_spec.txt 32B canonical row, LBO
// unused for K-major swizzled layouts).
WGEMU_HD bool sparse_b_layout_undefined(const Desc& d, bool mn_major) {
  return !mn_major && d.swizzle_bytes == 32;
}

WGEMU_HD uint32_t smem_addr(const Desc& d, bool mn_major, int elem_bits, int mn, int k) {
  const uint32_t eb = static_cast<uint32_t>(elem_bits >> 3);
  const uint32_t W = static_cast<uint32_t>(d.swizzle_bytes);
  uint32_t off;
  if (!mn_major) {
    const uint32_t r0 = static_cast<uint32_t>(mn) & 7u, r1 = static_cast<uint32_t>(mn) >> 3;
    const uint32_t kb = static_cast<uint32_t>(k) * eb;
    if (W == 0) {
      off = r0 * 16u + r1 * d.sbo + (kb & 15u) + (kb >> 4) * d.lbo;
    } else {
      off = r0 * W + r1 * d.sbo + kb;
    }
  } else {
    const uint32_t mb = static_cast<uint32_t>(mn) * eb;
    const uint32_t k0 = static_cast<uint32_t>(k) & 7u, k1 = static_cast<uint32_t>(k) >> 3;
    if (W == 0) {
      off = (mb & 15u) + (mb >> 4) * d.sbo + k0 * 16u + k1 * d.lbo;
    } else {
      off = (mb % W) + (mb / W) * d.lbo + k0 * W + k1 * d.sbo;
    }
  }
  return swizzle_addr(d.start + off, d.swizzle_bytes, d.base_off);
}

}  // namespace spec
}  // namespace wgemu
