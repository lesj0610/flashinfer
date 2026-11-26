// Direct GPU test of the wgmma device functions (no translator), PLAN section 1 item 3 and the
// check-mode part of item 5. Owner: W.
//
// Build (from the wgmma_emu root):
//   nvcc -arch=sm_80 -std=c++17 -O3 -I include tests/wgmma/test_direct.cu -o test_direct
// Run:
//   ./test_direct        -> WGEMU-TEST lines (PLAN section 11), WGEMU-TEST-END, exit 0 iff all pass
//
// Items:
//   wgmma.layout.fast_identity     wgmma RS A / D layouts (spec.h) vs the SM80 mma.sync fragment
//                                  literals the fast mode relies on, and the fp8 RS/SS mapping
//   wgmma.ref.literal_vs_spec      PLAN section 7 literals in ref_model.h vs spec.h
//   wgmma.fp8_to_f16               header conversion vs an independent double-based one
//   wgmma.variant_error            legal / illegal variants, encode/decode round trip
//   wgmma.layout.smem_injective    every operand element placed by the image builder (count of
//                                  elements; overlap / misalignment / out of range = mismatch)
//   wgmma.<mode>.<dense|sp>.<dt>.<at>.<bt>
//                                  one item per (mode, type combination); cases = D elements
//                                  compared bit for bit with ref_model.h (exact: Hopper wgmma
//                                  parameters; fast: the documented Ampere mma.sync decomposition)
//   wgmma.<mode>.base_off          base_offset 1..7 with 32B / 64B / 128B swizzle (not valid for
//                                  no-swizzle, wgmma_spec.txt:860): the base_offset = 0 image
//                                  shifted by 128 * base_off bytes must give the same D
//   wgmma.check.<scenario>         check mode: positive scenarios (no violation, values exact) and
//                                  negative scenarios (the expected violation kinds, nothing else)
//
// Coverage per type combination (PLAN section 1 item 3): every valid N (flags cycled); at the
// smallest and the largest N a grid of swizzle {none, 32B, 64B, 128B} x major {K, MN (f16/bf16)}
// x {SS, RS}, each of imm-scale-a/b, scale-d, satfinite, sp-sel taking both values at both N;
// LBO/SBO with gaps in both nesting orders, K-major swizzled start offsets, all valid metadata
// nibbles in every sparse case, random filler in every unused smem byte and metadata bit.
// Not generated: sparse B, K-major, 32B swizzle. Sparse B rows hold 64 bytes of K, wider than the
// 32-byte swizzle span; spec.h's [inferred] linear continuation (spec::smem_addr) gives element
// (row r, byte kb >= 32) the address of (row r + 1, byte kb - 32), so no operand image can hold
// both and the case cannot be tested against anything. Such B operands use 64B swizzle (the
// case name's `what` field shows the swizzle actually used).
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

#include "../../runtime/lib_wgmma.cu"
#include "ref_model.h"

namespace {

using wgemu::EType;
namespace wg = wgemu::wg;
namespace spec = wgemu::spec;

#define CK(x)                                                                               \
  do {                                                                                      \
    cudaError_t e_ = (x);                                                                   \
    if (e_ != cudaSuccess) {                                                                \
      printf("WGEMU-ERROR cuda %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      fflush(stdout);                                                                       \
      exit(2);                                                                              \
    }                                                                                       \
  } while (0)

std::map<std::string, wgref::Stat> g_stats;

wgref::Stat& stat_of(const std::string& name) { return g_stats[name]; }

void check_eq(const char* name, bool ok, const char* fmt_what, long a, long b) {
  wgref::Stat& s = stat_of(name);
  ++s.cases;
  if (ok) return;
  ++s.mismatches;
  if (s.printed < 8) {
    ++s.printed;
    printf("WGEMU-MISMATCH name=%s %s a=%ld b=%ld\n", name, fmt_what, a, b);
  }
}

// =============================================================================================
// CPU items
// =============================================================================================

// SM80 mma.sync fragment literals (PTX ISA "Matrix Fragments for mma.m16n8k*").
// A register r, element e -> (row within 16, column).
void mma_a_literal(EType at, int lane, int r, int e, int* row, int* col) {
  const int g = lane >> 2, q = lane & 3;
  *row = g + 8 * (r & 1);
  switch (wgref::type_bits(at)) {
    case 16:  // m16n8k16
      *col = 2 * q + e + 8 * (r >> 1);
      break;
    case 32:  // tf32 m16n8k8
      *col = q + 4 * (r >> 1);
      break;
    case 8:  // m16n8k32
      *col = 4 * q + e + 16 * (r >> 1);
      break;
    default:  // b1 m16n8k256, e = bit
      *col = 32 * q + e + 128 * (r >> 1);
      break;
  }
}

void cpu_fast_identity() {
  const char* name = "wgmma.layout.fast_identity";
  const EType types[] = {EType::F16, EType::BF16, EType::TF32, EType::S8, EType::B1};
  for (EType at : types) {
    for (int t = 0; t < 128; ++t) {
      const int w = t >> 5, lane = t & 31;
      for (int r = 0; r < 4; ++r)
        for (int e = 0; e < wgref::a_per_reg(at); ++e) {
          int m, k, row, col;
          spec::a_frag_coord(at, t, r, e, &m, &k);
          mma_a_literal(at, lane, r, e, &row, &col);
          check_eq(name, m == 16 * w + row && k == col, "a-frag", long(m) * 1000 + k,
                   long(16 * w + row) * 1000 + col);
        }
      // D (any N): n-tile j, element ii of the m16n8 C fragment.
      for (int i = 0; i < 128; ++i) {
        int m, n;
        spec::d_frag_coord(t, i, &m, &n);
        const int j = i / 4, ii = i % 4, g = lane >> 2, q = lane & 3;
        const int row = 16 * w + g + 8 * (ii >> 1), col = 8 * j + 2 * q + (ii & 1);
        check_eq(name, m == row && n == col, "d-frag", long(m) * 1000 + n, long(row) * 1000 + col);
      }
    }
  }
  // fp8 RS registers -> f16 mma fragment (wgmma_device.cuh fast mode) vs the SS getter mapping and
  // the column -> k map used by the fast reference.
  for (int t = 0; t < 128; ++t) {
    const int w = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
    for (int s = 0; s < 2; ++s)
      for (int r = 0; r < 4; ++r)
        for (int e2 = 0; e2 < 2; ++e2) {
          int m, k;
          spec::a_frag_coord(EType::E4M3, t, 2 * s + (r & 1), 2 * (r >> 1) + e2, &m, &k);
          const int mss = 16 * w + g + 8 * (r & 1), kss = 16 * s + 4 * q + 2 * (r >> 1) + e2;
          check_eq(name, m == mss && k == kss, "fp8-rs-vs-ss", long(m) * 1000 + k,
                   long(mss) * 1000 + kss);
          const int c = 2 * q + e2 + 8 * (r >> 1);  // f16 m16n8k16 column of this element
          check_eq(name, wgref::fast_kmap(EType::E4M3, s, c) == kss, "fp8-kmap",
                   wgref::fast_kmap(EType::E4M3, s, c), kss);
        }
  }
}

void cpu_literal_vs_spec() {
  const char* name = "wgmma.ref.literal_vs_spec";
  const EType types[] = {EType::F16,  EType::BF16, EType::TF32, EType::E4M3,
                         EType::E5M2, EType::S8,   EType::U8,   EType::B1};
  for (int t = 0; t < 128; ++t)
    for (int i = 0; i < 128; ++i) {
      int m0, n0, m1, n1;
      wgref::d_coord(t, i, &m0, &n0);
      spec::d_frag_coord(t, i, &m1, &n1);
      check_eq(name, m0 == m1 && n0 == n1, "d", long(m0) * 1000 + n0, long(m1) * 1000 + n1);
    }
  for (EType at : types) {
    for (int sp = 0; sp < 2; ++sp) {
      const bool sparse = sp != 0;
      if (sparse && at == EType::B1) continue;
      check_eq(name, wgref::kdim(at, sparse) == spec::k_of(at, sparse), "k",
               wgref::kdim(at, sparse), spec::k_of(at, sparse));
      for (int t = 0; t < 128; ++t)
        for (int j = 0; j < 4; ++j)
          for (int e = 0; e < wgref::a_per_reg(at); ++e) {
            int m0, k0, m1, k1;
            wgref::a_coord(at, sparse, t, j, e, &m0, &k0);
            if (sparse)
              spec::sp_a_frag_coord(at, t, j, e, &m1, &k1);
            else
              spec::a_frag_coord(at, t, j, e, &m1, &k1);
            check_eq(name, m0 == m1 && k0 == k1, sparse ? "sp-a" : "a", long(m0) * 1000 + k0,
                     long(m1) * 1000 + k1);
          }
      if (!sparse) continue;
      check_eq(name, wgref::meta_chunks(at) == spec::sp_meta_chunks(at), "chunks",
               wgref::meta_chunks(at), spec::sp_meta_chunks(at));
      const int nsel = wgref::type_bits(at) == 8 ? 1 : 2;
      for (int sel = 0; sel < nsel; ++sel)
        for (int m = 0; m < 64; ++m)
          for (int c = 0; c < wgref::meta_chunks(at); ++c) {
            int l0, b0, l1, b1;
            wgref::meta_loc(at, sel, m, c, &l0, &b0);
            spec::sp_meta_loc(at, sel, m, c, &l1, &b1);
            check_eq(name, l0 == l1 && b0 == b1, "meta-loc", long(l0) * 100 + b0,
                     long(l1) * 100 + b1);
          }
      for (uint32_t f = 0; f < 16; ++f) {
        int i0, i1;
        spec::sp_meta_decode(at, f, &i0, &i1);
        const bool ref_valid =
            at == EType::TF32 ? (f == 0x4u || f == 0xEu) : ((f & 3u) != (f >> 2));
        check_eq(name, spec::sp_meta_valid(at, f) == ref_valid, "meta-valid",
                 spec::sp_meta_valid(at, f), ref_valid);
        if (!ref_valid) continue;
        const int r0 = at == EType::TF32 ? int((f & 3u) >> 1) : int(f & 3u);
        const int r1 = at == EType::TF32 ? -1 : int(f >> 2);
        check_eq(name, i0 == r0 && i1 == r1, "meta-decode", i0 * 10 + i1, r0 * 10 + r1);
      }
    }
    for (int n = 0; n <= 264; n += 4)
      check_eq(name, wgref::n_valid(at, n) == spec::valid_n(at, n), "valid-n",
               wgref::n_valid(at, n) * 1000 + n, spec::valid_n(at, n) * 1000 + n);
  }
}

void cpu_fp8() {
  const char* name = "wgmma.fp8_to_f16";
  for (int ti = 0; ti < 2; ++ti) {
    const EType t = ti ? EType::E5M2 : EType::E4M3;
    for (uint32_t x = 0; x < 256; ++x) {
      const uint32_t a = wg::fp8_to_f16(x, t), b = wgref::fp8_to_f16_ref(x, t);
      check_eq(name, a == b, ti ? "e5m2" : "e4m3", long(x) << 16 | a, long(x) << 16 | b);
    }
  }
}

void cpu_variant_error() {
  const char* name = "wgmma.variant_error";
  struct Bad {
    int n;
    EType at, bt, dt;
    bool rs, sparse, neg_a, ta, tb, sat;
    int sel;
  };
  const Bad bad[] = {
      {8, EType::BF16, EType::BF16, EType::F16, false, false, false, false, false, false, 0},
      {8, EType::TF32, EType::TF32, EType::F16, false, false, false, false, false, false, 0},
      {8, EType::F16, EType::BF16, EType::F32, false, false, false, false, false, false, 0},
      {8, EType::E4M3, EType::E4M3, EType::S32, false, false, false, false, false, false, 0},
      {8, EType::S8, EType::S8, EType::F32, false, false, false, false, false, false, 0},
      {8, EType::S8, EType::E4M3, EType::S32, false, false, false, false, false, false, 0},
      {8, EType::B1, EType::B1, EType::S32, false, true, false, false, false, false, 0},
      {40, EType::S8, EType::S8, EType::S32, false, false, false, false, false, false, 0},
      {40, EType::B1, EType::B1, EType::S32, false, false, false, false, false, false, 0},
      {264, EType::F16, EType::F16, EType::F32, false, false, false, false, false, false, 0},
      {8, EType::TF32, EType::TF32, EType::F32, false, false, false, true, false, false, 0},
      {8, EType::E4M3, EType::E4M3, EType::F32, false, false, false, false, true, false, 0},
      {8, EType::F16, EType::F16, EType::F32, true, false, false, true, false, false, 0},
      {8, EType::S8, EType::S8, EType::S32, false, false, true, false, false, false, 0},
      {8, EType::F16, EType::F16, EType::F32, false, false, false, false, false, true, 0},
      {8, EType::F16, EType::F16, EType::F32, false, false, false, false, false, false, 1},
      {8, EType::E4M3, EType::E4M3, EType::F32, false, true, false, false, false, false, 1},
      {8, EType::S8, EType::U8, EType::S32, false, true, false, false, false, false, 1},
      {8, EType::F16, EType::F16, EType::F32, false, true, false, false, false, false, 2},
  };
  for (const Bad& b : bad) {
    wg::Variant v = wg::make_variant(b.n, b.at, b.bt, b.dt);
    v.rs = b.rs;
    v.sparse = b.sparse;
    v.neg_a = b.neg_a;
    v.trans_a = b.ta;
    v.trans_b = b.tb;
    v.satfinite = b.sat;
    v.sp_sel = b.sel;
    check_eq(name, wg::variant_error(v) != nullptr, "illegal-accepted", b.n, int(b.at));
  }
  wg::Variant v = wg::make_variant(8, EType::F16, EType::F16, EType::F32);
  const uint32_t x = wg::encode_variant(v);
  check_eq(name, wg::variant_error(wg::decode_variant(x | wg::VAR_RESERVED)) != nullptr,
           "reserved-accepted", 0, 0);
  check_eq(name, wg::variant_error(wg::decode_variant(x | (3u << wg::VAR_MODE_SHIFT))) != nullptr,
           "mode3-accepted", 0, 0);
}

// =============================================================================================
// GPU exact / fast cases
// =============================================================================================
constexpr uint32_t REG = 163840;             // smem region of the (single) warpgroup of a block
constexpr uint32_t SMEM_BYTES = 1024 + REG;  // + 1024 for the 1024-byte alignment pad
constexpr int IN_WORDS = 5;                  // a0..a3, meta, then C registers

struct DevCase {
  uint64_t adesc, bdesc;  // region-relative start addresses
  uint64_t img_off;       // bytes, 16-aligned
  uint64_t in_off, out_off;
  uint32_t img_bytes, variant, scale_d, nregs, ids, pad;
};

// One case per block (one warpgroup).
__global__ void __launch_bounds__(128)
    direct_kernel(const DevCase* cs, const uint8_t* img, const uint32_t* rin, uint32_t* rout) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const uint32_t sh0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  const uint32_t pad = ((sh0 + 1023u) & ~1023u) - sh0;
  const int t = int(threadIdx.x);
  const int ci = int(blockIdx.x);
  const DevCase c = cs[ci];
  uint8_t* region = smem_raw + pad;
  const uint32_t region_sh = sh0 + pad;
  const uint4* src = reinterpret_cast<const uint4*>(img + c.img_off);
  uint4* dst = reinterpret_cast<uint4*>(region);
  const uint32_t nimg = c.img_bytes / 16u;
  for (uint32_t i = uint32_t(t); i < REG / 16u; i += 128u) {
    uint4 v;
    if (i < nimg) {
      v = src[i];
    } else {
      const uint32_t h = (i * 2654435761u) ^ (uint32_t(ci) * 0x9E3779B9u) ^ 0xA5A5A5A5u;
      v = make_uint4(h, ~h, h * 3u, h ^ 0x5A5A5A5Au);
    }
    dst[i] = v;
  }
  __syncthreads();
  const uint64_t base4 = region_sh >> 4;
  const bool rs = (c.variant & wg::VAR_RS) != 0u;
  if (((c.bdesc & 0x3FFFu) + base4) > 0x3FFFu || (!rs && ((c.adesc & 0x3FFFu) + base4) > 0x3FFFu))
    __trap();
  const uint64_t adesc = rs ? 0ull : c.adesc + base4;
  const uint64_t bdesc = c.bdesc + base4;
  const uint32_t* in = rin + c.in_off + uint64_t(t) * (IN_WORDS + c.nregs);
  uint32_t d[128];
  for (uint32_t i = 0; i < c.nregs; ++i) d[i] = in[IN_WORDS + i];
  wgemu_wgmma(reinterpret_cast<uint64_t>(d), adesc, in[0], in[1], in[2], in[3], bdesc, in[4],
              c.scale_d, c.variant, c.ids, 0ull);
  uint32_t* out = rout + c.out_off + uint64_t(t) * c.nregs;
  for (uint32_t i = 0; i < c.nregs; ++i) out[i] = d[i];
}

struct Combo {
  EType at, bt, dt;
  bool sparse;
};

std::vector<Combo> all_combos() {
  std::vector<Combo> v;
  const EType fp8[] = {EType::E4M3, EType::E5M2}, i8[] = {EType::S8, EType::U8};
  for (int sp = 0; sp < 2; ++sp) {
    const bool s = sp != 0;
    v.push_back({EType::F16, EType::F16, EType::F16, s});
    v.push_back({EType::F16, EType::F16, EType::F32, s});
    v.push_back({EType::BF16, EType::BF16, EType::F32, s});
    v.push_back({EType::TF32, EType::TF32, EType::F32, s});
    for (EType a : fp8)
      for (EType b : fp8) {
        v.push_back({a, b, EType::F16, s});
        v.push_back({a, b, EType::F32, s});
      }
    for (EType a : i8)
      for (EType b : i8) v.push_back({a, b, EType::S32, s});
    if (!s) v.push_back({EType::B1, EType::B1, EType::S32, false});
  }
  return v;
}

struct Case {
  std::string name, what;
  wgref::Op op;
  bool rs = false, trans_a = false, trans_b = false;
  wgref::LayoutSel la, lb;
  int base_off = 0;
  bool fast = false;
  uint64_t seed = 0;
};

const int SW[4] = {0, 32, 64, 128};

bool trans_ok(EType at) { return at == EType::F16 || at == EType::BF16; }
bool sel_ok(const wgref::Op& o) {
  return o.sparse && (wgref::type_bits(o.at) == 16 || o.at == EType::TF32);
}

void finish_case(Case& c) {
  c.la.mn_major = c.trans_a;
  c.lb.mn_major = c.trans_b;
  if (c.op.sparse && !c.lb.mn_major && c.lb.sw == 32) c.lb.sw = 64;  // see header comment
  char buf[256];
  snprintf(buf, sizeof(buf),
           "n=%d rs=%d ta=%d tb=%d na=%d nb=%d sd=%d sat=%d sel=%d swA=%d swB=%d oA=%d oB=%d "
           "kA=%d kB=%d gA=%d gB=%d boff=%d seed=%llu",
           c.op.n, c.rs, c.trans_a, c.trans_b, c.op.neg_a, c.op.neg_b, c.op.scale_d, c.op.satfinite,
           c.op.sp_sel, c.la.sw, c.lb.sw, c.la.order, c.lb.order, c.la.kofs, c.lb.kofs, c.la.gap,
           c.lb.gap, c.base_off, (unsigned long long)c.seed);
  c.what = buf;
}

uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

std::vector<Case> build_cases() {
  std::vector<Case> out;
  uint64_t seed = 1;
  std::vector<Case> base;
  const std::vector<Combo> combos = all_combos();
  for (size_t ci = 0; ci < combos.size(); ++ci) {
    const Combo& cb = combos[ci];
    const bool fl = wgref::is_float_in(cb.at), i8 = cb.at == EType::S8 || cb.at == EType::U8;
    char nm[96];
    snprintf(nm, sizeof(nm), "%s.%s.%s.%s", cb.sparse ? "sp" : "dense", wgref::type_name(cb.dt),
             wgref::type_name(cb.at), wgref::type_name(cb.bt));
    const std::vector<int> ns = wgref::all_n(cb.at);
    auto mk = [&](int n) {
      Case c;
      c.name = nm;
      c.op.n = n;
      c.op.at = cb.at;
      c.op.bt = cb.bt;
      c.op.dt = cb.dt;
      c.op.sparse = cb.sparse;
      c.seed = seed++;
      return c;
    };
    // N sweep with cycled flags.
    for (size_t idx = 0; idx < ns.size(); ++idx) {
      Case c = mk(ns[idx]);
      const uint32_t h = mix32(uint32_t(idx * 0x9E3779B1u + ci * 7919u + 1u));
      c.rs = (idx & 1) != 0;
      c.trans_a = trans_ok(cb.at) && !c.rs && ((idx >> 1) & 1);
      c.trans_b = trans_ok(cb.at) && ((idx >> 2) & 1);
      c.op.neg_a = fl && (h & 1u);
      c.op.neg_b = fl && ((h >> 1) & 1u);
      c.op.scale_d = ((h >> 2) & 3u) != 0u;
      c.op.satfinite = i8 && ((h >> 4) & 1u);
      c.op.sp_sel = sel_ok(c.op) ? int((h >> 5) & 1u) : 0;
      c.la.sw = SW[idx % 4];
      c.lb.sw = SW[(idx + 1 + idx / 4) % 4];
      c.la.order = int((h >> 6) & 1u);
      c.lb.order = int((h >> 7) & 1u);
      c.la.kofs = int((h >> 8) & 3u);
      c.lb.kofs = int((h >> 10) & 3u);
      c.la.gap = int((h >> 12) & 3u);
      c.lb.gap = int((h >> 14) & 3u);
      finish_case(c);
      base.push_back(c);
    }
    // Grid at the smallest and the largest N.
    const int grid_n[2] = {ns.front(), ns.back()};
    for (int gi = 0; gi < 2; ++gi) {
      int k = 0;
      for (int sw = 0; sw < 4; ++sw)
        for (int maj = 0; maj < (trans_ok(cb.at) ? 2 : 1); ++maj)
          for (int rs = 0; rs < 2; ++rs, ++k) {
            Case c = mk(grid_n[gi]);
            c.rs = rs != 0;
            c.trans_a = maj && !c.rs;
            c.trans_b = maj != 0;
            c.op.neg_a = fl && ((k >> 1) & 1);
            c.op.neg_b = fl && (((k >> 1) ^ (k >> 2)) & 1);
            c.op.scale_d = ((k ^ (k >> 2)) & 1) != 0;
            c.op.satfinite = i8 && ((k >> 2) & 1);
            c.op.sp_sel = sel_ok(c.op) ? (((k >> 1) ^ k) & 1) : 0;
            c.la.sw = c.lb.sw = SW[sw];
            c.la.order = (k >> 1) & 1;
            c.lb.order = k & 1;
            c.la.kofs = k;
            c.lb.kofs = k + 1;
            c.la.gap = k & 3;
            c.lb.gap = (k >> 1) & 3;
            finish_case(c);
            base.push_back(c);
          }
    }
  }
  // base_offset != 0.
  const Combo bo[] = {{EType::F16, EType::F16, EType::F32, false},
                      {EType::E4M3, EType::E5M2, EType::F16, false},
                      {EType::F16, EType::F16, EType::F32, true},
                      {EType::S8, EType::U8, EType::S32, false}};
  for (const Combo& cb : bo)
    for (int sw = 1; sw < 4; ++sw)  // base_offset is not valid for no-swizzle (PTX 9.7.17.5.1.2)
      for (int b = 1; b <= 7; ++b) {
        Case c;
        c.name = "base_off";
        c.op.n = 64;
        c.op.at = cb.at;
        c.op.bt = cb.bt;
        c.op.dt = cb.dt;
        c.op.sparse = cb.sparse;
        c.op.neg_a = wgref::is_float_in(cb.at) && (b & 2);
        c.op.scale_d = (b & 1) != 0;
        c.trans_a = c.trans_b = trans_ok(cb.at) && (b & 1);
        c.la.sw = c.lb.sw = SW[sw];
        c.la.order = b & 1;
        c.lb.order = (b >> 1) & 1;
        c.la.kofs = b;
        c.lb.kofs = b + 1;
        c.base_off = b;
        c.seed = seed++;
        finish_case(c);
        base.push_back(c);
      }
  for (int mode = 0; mode < 2; ++mode)
    for (const Case& b : base) {
      Case c = b;
      c.fast = mode == 1;
      c.name = std::string("wgmma.") + (c.fast ? "fast." : "exact.") + b.name;
      out.push_back(c);
    }
  return out;
}

uint32_t make_var(const wgref::Op& o, bool rs, bool ta, bool tb, wg::Mode mode) {
  wg::Variant v = wg::make_variant(o.n, o.at, o.bt, o.dt);
  v.rs = rs;
  v.sparse = o.sparse;
  v.neg_a = o.neg_a;
  v.neg_b = o.neg_b;
  v.trans_a = ta;
  v.trans_b = tb;
  v.satfinite = o.satfinite;
  v.sp_sel = o.sp_sel;
  v.mode = mode;
  return wg::encode_variant(v);
}

struct Prepared {
  const Case* c;
  wgref::Problem P;
  std::vector<uint32_t> want;
  bool ok;
};

void run_gpu_cases() {
  const std::vector<Case> cases = build_cases();
  wgref::Image inj;  // accumulates the injectivity statistics
  const size_t CHUNK = 192;
  for (size_t c0 = 0; c0 < cases.size(); c0 += CHUNK) {
    const size_t c1 = std::min(cases.size(), c0 + CHUNK);
    std::vector<DevCase> dcs;
    std::vector<uint8_t> imgs;
    std::vector<uint32_t> rin;
    std::vector<Prepared> prep;
    size_t out_words = 0;
    for (size_t i = c0; i < c1; ++i) {
      const Case& c = cases[i];
      Prepared pr;
      pr.c = &c;
      pr.ok = false;
      pr.P = wgref::make_problem(c.op, c.seed);
      std::vector<uint8_t> img;
      wgref::Placement pl;
      std::string why;
      if (!wgref::place_problem(pr.P, c.rs, c.la, c.lb, c.base_off, REG, c.seed, img, &pl, &inj,
                                &why)) {
        check_eq(c.name.c_str(), false, ("layout-error " + why + " " + c.what).c_str(), 0, 0);
        continue;
      }
      const uint32_t var =
          make_var(c.op, c.rs, c.trans_a, c.trans_b, c.fast ? wg::Mode::Fast : wg::Mode::Exact);
      if (const char* err = wg::variant_error(wg::decode_variant(var))) {
        check_eq(c.name.c_str(), false, (std::string("bad-variant ") + err + " " + c.what).c_str(),
                 0, 0);
        continue;
      }
      pr.ok = true;
      DevCase dc = {};
      dc.adesc = pl.adesc;
      dc.bdesc = pl.bdesc;
      dc.img_off = imgs.size();
      dc.img_bytes = pl.bytes;
      dc.variant = var;
      dc.scale_d = c.op.scale_d ? 1u : 0u;
      dc.nregs = uint32_t(wgref::op_nregs(c.op));
      dc.ids = wg::IDS_NO_A << 16;
      imgs.insert(imgs.end(), img.begin(), img.end());
      imgs.resize(wgref::align_up(uint32_t(imgs.size()), 16u), 0);
      dc.in_off = rin.size();
      dc.out_off = out_words;
      out_words += size_t(128) * dc.nregs;
      wgref::Rng r(c.seed ^ 0xC0FFEEull);
      std::vector<uint32_t> garbage(size_t(64) * c.op.n);
      for (auto& x : garbage) x = r.u32();
      const std::vector<uint32_t>& cm = c.op.scale_d ? pr.P.C : garbage;
      for (int t = 0; t < 128; ++t) {
        uint32_t a[4] = {r.u32(), r.u32(), r.u32(), r.u32()};
        if (c.rs) wgref::pack_a(pr.P, t, a);
        const uint32_t meta = c.op.sparse ? wgref::pack_meta(pr.P, t, r.u32()) : r.u32();
        uint32_t regs[128];
        wgref::pack_c(c.op, cm, t, regs);
        rin.insert(rin.end(), a, a + 4);
        rin.push_back(meta);
        rin.insert(rin.end(), regs, regs + dc.nregs);
      }
      pr.want = wgref::ref_mode(pr.P, pr.P.C, c.fast);
      dcs.push_back(dc);
      prep.push_back(std::move(pr));
    }
    if (dcs.empty()) continue;
    DevCase* d_cs;
    uint8_t* d_img;
    uint32_t *d_in, *d_out;
    CK(cudaMalloc(&d_cs, dcs.size() * sizeof(DevCase)));
    CK(cudaMalloc(&d_img, std::max<size_t>(imgs.size(), 16)));
    CK(cudaMalloc(&d_in, rin.size() * 4));
    CK(cudaMalloc(&d_out, out_words * 4));
    CK(cudaMemcpy(d_cs, dcs.data(), dcs.size() * sizeof(DevCase), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_img, imgs.data(), imgs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_in, rin.data(), rin.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0xCD, out_words * 4));
    direct_kernel<<<int(dcs.size()), 128, SMEM_BYTES>>>(d_cs, d_img, d_in, d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> outv(out_words);
    CK(cudaMemcpy(outv.data(), d_out, out_words * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_cs));
    CK(cudaFree(d_img));
    CK(cudaFree(d_in));
    CK(cudaFree(d_out));
    size_t k = 0;
    for (const Prepared& pr : prep) {
      const DevCase& dc = dcs[k++];
      const Case& c = *pr.c;
      std::vector<uint32_t> got(size_t(64) * c.op.n, 0u);
      for (int t = 0; t < 128; ++t)
        wgref::unpack_d(c.op, t, outv.data() + dc.out_off + size_t(t) * dc.nregs, got);
      wgref::compare(c.name.c_str(), c.what, c.op, got, pr.want, stat_of(c.name));
    }
  }
  wgref::Stat& s = stat_of("wgmma.layout.smem_injective");
  s.cases += inj.injective_checked;
  s.mismatches += inj.injective_bad;
}

// =============================================================================================
// GPU check-mode scenarios
// =============================================================================================
constexpr uint32_t TAB_BYTES = wg::WG_HDR_BYTES + 2u * wg::WG_ACC_SLOT_BYTES + wg::WG_A_SLOT_BYTES +
                               wg::WG_MAX_OPS * wg::WG_OP_BYTES;
constexpr int CK_IN = 14;   // c[8], a[4], meta, meta_bad
constexpr int CK_OUT = 20;  // d0[8], d1[8], rets[4]

// Problems: 0 f16 SS (k16), 1 e4m3 SS (k32), 2 f16 sparse SS, 3 f16 RS. All N = 16, f32 D.
struct CheckArgs {
  const uint8_t* img;
  uint32_t img_bytes, flip_addr;
  uint64_t adesc[4], bdesc[4];
  uint32_t var[4];
  const uint32_t* rin;
  uint32_t* rout;
  uint32_t* kinds;
  uint32_t* viol;
};

enum Scn {
  POS_CHAIN,
  POS_SHAPE_FENCED,
  POS_RS,
  POS_SPARSE,
  POS_FENCED_AFTER_FETCH,
  POS_GUARD_FALSE,
  NEG_FENCE_FIRST,
  NEG_FENCE_AFTER_FETCH,
  NEG_ACC_MODIFIED,
  NEG_ACC_READ_POISON,
  NEG_ACC_ACCESS,
  NEG_SMEM_MODIFIED,
  NEG_RS_A_AT_WAIT,
  NEG_RS_A_AT_REISSUE,
  NEG_SP_META,
  NEG_WAIT_TOO_FEW,
  NEG_SHAPE_NO_FENCE,
  N_SCN
};
const char* const kScnName[N_SCN] = {"pos_chain",
                                     "pos_shape_change_fenced",
                                     "pos_rs",
                                     "pos_sparse",
                                     "pos_fenced_after_fetch",
                                     "pos_guard_false",
                                     "neg_fence_missing_first",
                                     "neg_fence_missing_after_fetch",
                                     "neg_acc_modified",
                                     "neg_acc_read_poison_c",
                                     "neg_acc_access_pending",
                                     "neg_smem_modified",
                                     "neg_rs_a_modified_at_wait",
                                     "neg_rs_a_modified_at_reissue",
                                     "neg_sp_meta_invalid",
                                     "neg_wait_too_few",
                                     "neg_shape_change_no_fence"};
const uint32_t kScnKinds[N_SCN] = {0u,
                                   0u,
                                   0u,
                                   0u,
                                   0u,
                                   0u,
                                   1u << wg::V_FENCE_MISSING,
                                   1u << wg::V_FENCE_MISSING,
                                   1u << wg::V_ACC_MODIFIED,
                                   1u << wg::V_ACC_READ,
                                   1u << wg::V_ACC_ACCESS,
                                   1u << wg::V_SMEM_MODIFIED,
                                   1u << wg::V_RS_A_MODIFIED,
                                   1u << wg::V_RS_A_MODIFIED,
                                   1u << wg::V_SP_META,
                                   1u << wg::V_ACC_ACCESS,
                                   1u << wg::V_FENCE_MISSING};

__global__ void __launch_bounds__(128) check_kernel(int sc, CheckArgs g) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const uint32_t sh0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  const uint32_t pad = ((sh0 + 1023u) & ~1023u) - sh0;
  uint8_t* region = smem_raw + pad;
  const uint64_t base4 = (sh0 + pad) >> 4;
  const int t = int(threadIdx.x);
  for (uint32_t i = uint32_t(t); i < g.img_bytes / 16u; i += 128u)
    reinterpret_cast<uint4*>(region)[i] = reinterpret_cast<const uint4*>(g.img)[i];
  __syncthreads();
  uint64_t ad[4], bd[4];
  for (int p = 0; p < 4; ++p) {
    ad[p] = p == 3 ? 0ull : g.adesc[p] + base4;
    bd[p] = g.bdesc[p] + base4;
  }
  alignas(16) uint8_t tab[TAB_BYTES];
  const uint64_t pend = reinterpret_cast<uint64_t>(tab);
  wgemu_wgmma_check_init(pend, 2u, 1u, 0u);
  wg::TabHeader* H = reinterpret_cast<wg::TabHeader*>(tab);
  H->printed = 2u;  // the direct test judges by kinds_seen; keep its output to WGEMU-TEST lines
  const uint32_t* in = g.rin + t * CK_IN;
  uint32_t d0[8], d1[8], a[4], rets[4] = {9u, 9u, 9u, 9u};
  for (int i = 0; i < 8; ++i) d0[i] = d1[i] = in[i];
  for (int j = 0; j < 4; ++j) a[j] = in[8 + j];
  uint32_t meta = in[12];
  const uint32_t ID0 = 0u | (wg::IDS_NO_A << 16), ID1 = 1u | (wg::IDS_NO_A << 16);
  const uint32_t IDRS0 = 0u, IDRS1 = 1u;  // accumulator 0 / 1 with A-register id 0
  auto mma = [&](uint32_t* d, int p, uint32_t sd, uint32_t ids) {
    wgemu_wgmma(reinterpret_cast<uint64_t>(d), ad[p], a[0], a[1], a[2], a[3], bd[p], meta, sd,
                g.var[p], ids, pend);
  };
  auto fetch = [&](uint32_t id, uint32_t* d) {
    return wgemu_wgmma_fetch(pend, id, reinterpret_cast<uint64_t>(d));
  };
  switch (sc) {
    case POS_CHAIN:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      mma(d1, 0, 0u, ID1);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 1u);
      rets[0] = fetch(0u, d0);
      rets[1] = fetch(1u, d1);
      wgemu_wgmma_wait(pend, 0u);
      rets[2] = fetch(1u, d1);
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[3] = fetch(0u, d0);
      break;
    case POS_SHAPE_FENCED:
    case NEG_SHAPE_NO_FENCE:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      if (sc == POS_SHAPE_FENCED) wgemu_wgmma_fence(pend);
      mma(d0, 1, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case POS_RS:
      wgemu_wgmma_fence(pend);
      mma(d0, 3, 1u, IDRS0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      wgemu_wgmma_check_a(pend, 0u, a[0], a[1], a[2], a[3]);
      break;
    case POS_SPARSE:
      wgemu_wgmma_fence(pend);
      mma(d0, 2, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case POS_FENCED_AFTER_FETCH:
    case NEG_FENCE_AFTER_FETCH:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      d0[0] ^= 0x00400000u;
      if (sc == POS_FENCED_AFTER_FETCH) wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[1] = fetch(0u, d0);
      break;
    case POS_GUARD_FALSE:
    case NEG_ACC_ACCESS:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_check_acc(pend, 0u, 0u);  // guarded instruction, guard false
      if (sc == NEG_ACC_ACCESS) wgemu_wgmma_check_acc(pend, 0u, 1u);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_FENCE_FIRST:
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_ACC_MODIFIED:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      d0[3] = 0x3F800000u;
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_ACC_READ_POISON:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      for (int i = 0; i < 8; ++i) d1[i] = d0[i];
      mma(d1, 0, 1u, ID1);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      rets[1] = fetch(1u, d1);
      break;
    case NEG_SMEM_MODIFIED:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      __syncthreads();
      if (t == 0) region[g.flip_addr] ^= 0x01u;
      __syncthreads();
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_RS_A_AT_WAIT:
      wgemu_wgmma_fence(pend);
      mma(d0, 3, 1u, IDRS0);
      wgemu_wgmma_commit(pend);
      a[1] ^= 0x00008000u;
      wgemu_wgmma_wait(pend, 0u);
      wgemu_wgmma_check_a(pend, 0u, a[0], a[1], a[2], a[3]);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_RS_A_AT_REISSUE:
      wgemu_wgmma_fence(pend);
      mma(d0, 3, 1u, IDRS0);
      wgemu_wgmma_commit(pend);
      a[2] ^= 0x00000001u;
      mma(d1, 3, 1u, IDRS1);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      rets[1] = fetch(1u, d1);
      break;
    case NEG_SP_META:
      meta = in[13];
      wgemu_wgmma_fence(pend);
      mma(d0, 2, 1u, ID0);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 0u);
      rets[0] = fetch(0u, d0);
      break;
    case NEG_WAIT_TOO_FEW:
      wgemu_wgmma_fence(pend);
      mma(d0, 0, 1u, ID0);
      wgemu_wgmma_commit(pend);
      mma(d1, 0, 1u, ID1);
      wgemu_wgmma_commit(pend);
      wgemu_wgmma_wait(pend, 1u);
      rets[0] = fetch(0u, d0);
      rets[1] = fetch(1u, d1);
      wgemu_wgmma_check_acc(pend, 1u, 1u);  // the program reads accumulator 1 here
      wgemu_wgmma_wait(pend, 0u);
      rets[2] = fetch(1u, d1);
      break;
    default:
      __trap();
  }
  uint32_t* out = g.rout + t * CK_OUT;
  for (int i = 0; i < 8; ++i) {
    out[i] = d0[i];
    out[8 + i] = d1[i];
  }
  for (int i = 0; i < 4; ++i) out[16 + i] = rets[i];
  atomicOr(g.kinds, H->kinds_seen);
  atomicAdd(g.viol, H->violations);
}

void run_check_tests() {
  wgref::Problem P[4];
  wgref::Op o;
  o.n = 16;
  o.dt = EType::F32;
  P[0] = wgref::make_problem(o, 9001);
  wgref::Op o1 = o;
  o1.at = o1.bt = EType::E4M3;
  P[1] = wgref::make_problem(o1, 9002);
  wgref::Op o2 = o;
  o2.sparse = true;
  o2.sp_sel = 1;
  P[2] = wgref::make_problem(o2, 9003);
  P[3] = wgref::make_problem(o, 9004);
  for (int p = 1; p < 4; ++p) P[p].C = P[0].C;
  const bool rs[4] = {false, false, false, true};
  wgref::LayoutSel la[4], lb[4];
  la[0].sw = 128;
  lb[0].sw = 128;
  la[1].sw = 64;
  lb[1].sw = 32;
  la[2].sw = 32;
  lb[2].sw = 128;
  lb[3].sw = 64;
  lb[3].mn_major = true;
  const bool tb[4] = {false, false, false, true};
  std::vector<uint8_t> img;
  CheckArgs g = {};
  uint32_t flip = 0;
  for (int p = 0; p < 4; ++p) {
    std::vector<uint8_t> part;
    wgref::Placement pl;
    std::string why;
    if (!wgref::place_problem(P[p], rs[p], la[p], lb[p], 0, 32768, 777 + p, part, &pl, nullptr,
                              &why)) {
      check_eq("wgmma.check.setup", false, why.c_str(), p, 0);
      return;
    }
    const uint32_t off = wgref::align_up(uint32_t(img.size()), 1024u);
    img.resize(off, 0);
    img.insert(img.end(), part.begin(), part.end());
    g.adesc[p] = rs[p] ? 0ull : pl.adesc + (off >> 4);
    g.bdesc[p] = pl.bdesc + (off >> 4);
    if (p == 0) flip = pl.b_first_addr + off;
    g.var[p] = make_var(P[p].op, rs[p], false, tb[p], wg::Mode::Check);
  }
  img.resize(wgref::align_up(uint32_t(img.size()), 16u), 0);
  g.img_bytes = uint32_t(img.size());
  g.flip_addr = flip;
  // Per-thread inputs.
  std::vector<uint32_t> rin(size_t(128) * CK_IN);
  wgref::Rng r(4242);
  int bad_lane, bad_bit;
  wgref::meta_loc(EType::F16, 1, 0, 0, &bad_lane, &bad_bit);  // row 0, chunk 0 (warp 0)
  for (int t = 0; t < 128; ++t) {
    uint32_t* w = rin.data() + size_t(t) * CK_IN;
    wgref::pack_c(o, P[0].C, t, w);
    wgref::pack_a(P[3], t, w + 8);
    w[12] = wgref::pack_meta(P[2], t, r.u32());
    w[13] = (t == bad_lane) ? (w[12] & ~(0xFu << bad_bit)) : w[12];
  }
  uint8_t* d_img;
  uint32_t *d_in, *d_out, *d_k;
  CK(cudaMalloc(&d_img, img.size()));
  CK(cudaMalloc(&d_in, rin.size() * 4));
  CK(cudaMalloc(&d_out, size_t(128) * CK_OUT * 4));
  CK(cudaMalloc(&d_k, 8));
  CK(cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_in, rin.data(), rin.size() * 4, cudaMemcpyHostToDevice));
  g.img = d_img;
  g.rin = d_in;
  g.rout = d_out;
  g.kinds = d_k;
  g.viol = d_k + 1;
  const size_t smem = 1024 + img.size();
  if (smem > 65536) {
    check_eq("wgmma.check.setup", false, "image larger than 64 KiB", long(smem), 65536);
    return;
  }
  // Expected values.
  wgref::Problem P0z = P[0];
  P0z.op.scale_d = false;
  const std::vector<uint32_t> e1 = wgref::ref_exact(P[0]);
  const std::vector<uint32_t> e2 = wgref::ref_exact(P[0], e1);
  const std::vector<uint32_t> chain3 = wgref::ref_exact(P[0], e2);
  const std::vector<uint32_t> d1z = wgref::ref_exact(P0z);
  const std::vector<uint32_t> shape = wgref::ref_exact(P[1], e1);
  const std::vector<uint32_t> rsr = wgref::ref_exact(P[3]);
  const std::vector<uint32_t> spr = wgref::ref_exact(P[2]);
  std::vector<uint32_t> e1f = e1;
  for (int t = 0; t < 128; ++t) {
    int m, n;
    wgref::d_coord(t, 0, &m, &n);
    e1f[size_t(m) * 16 + n] ^= 0x00400000u;
  }
  const std::vector<uint32_t> after = wgref::ref_exact(P[0], e1f);
  for (int sc = 0; sc < N_SCN; ++sc) {
    const std::string name = std::string("wgmma.check.") + kScnName[sc];
    CK(cudaMemset(d_k, 0, 8));
    CK(cudaMemset(d_out, 0xCD, size_t(128) * CK_OUT * 4));
    check_kernel<<<1, 128, smem>>>(sc, g);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    uint32_t kv[2];
    std::vector<uint32_t> outv(size_t(128) * CK_OUT);
    CK(cudaMemcpy(kv, d_k, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outv.data(), d_out, outv.size() * 4, cudaMemcpyDeviceToHost));
    check_eq(name.c_str(), kv[0] == kScnKinds[sc], "kinds_seen", kv[0], kScnKinds[sc]);
    if (kScnKinds[sc] == 0u)
      check_eq(name.c_str(), kv[1] == 0u, "violations", kv[1], 0);
    else
      check_eq(name.c_str(), kv[1] > 0u, "violations", kv[1], 1);
    if (sc == NEG_ACC_ACCESS) check_eq(name.c_str(), kv[1] == 128u, "one-per-thread", kv[1], 128);
    std::vector<uint32_t> d0(size_t(64) * 16), d1(size_t(64) * 16);
    std::vector<uint32_t> rets(size_t(128) * 4);
    for (int t = 0; t < 128; ++t) {
      const uint32_t* w = outv.data() + size_t(t) * CK_OUT;
      wgref::unpack_d(o, t, w, d0);
      wgref::unpack_d(o, t, w + 8, d1);
      for (int i = 0; i < 4; ++i) rets[size_t(t) * 4 + i] = w[16 + i];
    }
    auto want_rets = [&](std::initializer_list<uint32_t> v) {
      for (int t = 0; t < 128; ++t) {
        int i = 0;
        for (uint32_t x : v) {
          check_eq(name.c_str(), rets[size_t(t) * 4 + i] == x, "fetch-ret", rets[size_t(t) * 4 + i],
                   x);
          ++i;
        }
      }
    };
    wgref::Stat& st = stat_of(name);
    switch (sc) {
      case POS_CHAIN:
        wgref::compare(name.c_str(), "d0", o, d0, chain3, st);
        wgref::compare(name.c_str(), "d1", o, d1, d1z, st);
        want_rets({1u, 0u, 1u, 1u});
        break;
      case POS_SHAPE_FENCED:
        wgref::compare(name.c_str(), "d0", o, d0, shape, st);
        want_rets({1u});
        break;
      case POS_RS:
        wgref::compare(name.c_str(), "d0", o, d0, rsr, st);
        want_rets({1u});
        break;
      case POS_SPARSE:
        wgref::compare(name.c_str(), "d0", o, d0, spr, st);
        want_rets({1u});
        break;
      case POS_FENCED_AFTER_FETCH:
        wgref::compare(name.c_str(), "d0", o, d0, after, st);
        want_rets({1u, 1u});
        break;
      case POS_GUARD_FALSE:
        wgref::compare(name.c_str(), "d0", o, d0, e1, st);
        want_rets({1u});
        break;
      case NEG_WAIT_TOO_FEW:
        want_rets({1u, 0u, 1u});
        break;
      default:
        break;
    }
  }
  CK(cudaFree(d_img));
  CK(cudaFree(d_in));
  CK(cudaFree(d_out));
  CK(cudaFree(d_k));
}

}  // namespace

int main() {
  cpu_fast_identity();
  cpu_literal_vs_spec();
  cpu_fp8();
  cpu_variant_error();
  CK(cudaFuncSetAttribute(direct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          int(SMEM_BYTES)));
  CK(cudaFuncSetAttribute(check_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
  run_gpu_cases();
  run_check_tests();
  bool ok = true;
  for (const auto& kv : g_stats) {
    printf("WGEMU-TEST name=%s cases=%llu mismatches=%llu\n", kv.first.c_str(),
           (unsigned long long)kv.second.cases, (unsigned long long)kv.second.mismatches);
    if (kv.second.cases == 0 || kv.second.mismatches != 0) ok = false;
  }
  printf("WGEMU-TEST-END\n");
  return ok ? 0 : 1;
}
