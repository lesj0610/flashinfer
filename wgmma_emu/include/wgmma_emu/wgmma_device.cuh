// SM90 wgmma emulation on SM80 (fork-only). Owner: W.
//
// ============================================================================================
// Entry points (defined in runtime/lib_wgmma.cu; extern "C" __device__ __noinline__,
// u32/u64 scalar arguments only, pointers are generic u64):
//
//   void wgemu_wgmma(u64 d_ptr, u64 adesc, u32 a0, u32 a1, u32 a2, u32 a3, u64 bdesc,
//                    u32 meta, u32 scale_d, u32 variant, u32 ids, u64 pend)
//       One wgmma.mma_async{.sp} for the calling thread. d_ptr points at the thread's
//       accumulator image: nregs = N/2 (f32/s32 D) or N/4 (f16 D) u32 words, register i of
//       the PTX d vector at word i. The words hold C on entry and D on return.
//       SS: adesc is the A descriptor and a0..a3 are ignored. RS: a0..a3 are the four A
//       registers and adesc is ignored. meta is sp-meta (ignored when dense). scale_d != 0
//       means D = A*B + C, 0 means D = A*B. variant: see the table below.
//       ids: bits 0..15 accumulator id, bits 16..31 RS A-register id (0xFFFF = none);
//       used by check mode only. pend: check-mode table (generic pointer), 0 otherwise.
//   void wgemu_wgmma_check_init(u64 pend, u32 n_acc, u32 n_a, u32 flags)
//       Zero and initialise the check-mode table. flags bit 0: __trap() on violation.
//   void wgemu_wgmma_fence(u64 pend)            wgmma.fence.sync.aligned (check mode)
//   void wgemu_wgmma_commit(u64 pend)           wgmma.commit_group.sync.aligned (check mode)
//   void wgemu_wgmma_wait(u64 pend, u32 n)      wgmma.wait_group.sync.aligned n (check mode)
//   u32  wgemu_wgmma_fetch(u64 pend, u32 acc_id, u64 d_ptr)
//       After a wait. d_ptr holds the current accumulator registers on entry. Returns 1 and
//       writes the completed result to d_ptr when the accumulator finished in the last wait;
//       the translator then reloads the registers from d_ptr. Returns 0 otherwise.
//   void wgemu_wgmma_check_a(u64 pend, u32 a_id, u32 a0, u32 a1, u32 a2, u32 a3)
//       After a wait: current RS A registers of a_id, compared with the values at issue.
//   void wgemu_wgmma_check_acc(u64 pend, u32 acc_id, u32 kind)
//       Before a non-wgmma instruction that names accumulator registers. kind bit 0: the
//       instruction executes (1 when unguarded, else the value of its guard). Reports an
//       access while the accumulator is still pending.
//
// In exact and fast modes the translator emits only wgemu_wgmma (pend = 0); fence, commit and
// wait become no-ops because results are written at issue time.
//
// ============================================================================================
// variant (u32) bit layout. Keep identical to tools/ptxemu/rules_wgmma.py (VARIANT_* there).
//   bits  0..5   N/8 - 1            (N = 8..256)
//   bits  6..9   atype              (wgemu::EType value)
//   bits 10..13  btype              (wgemu::EType value)
//   bits 14..17  dtype              (wgemu::EType value: F16=0, F32=8, S32=9)
//   bit  18      rs                 A from registers a0..a3 (else from adesc)
//   bit  19      sparse             wgmma.mma_async.sp
//   bit  20      neg_a              imm-scale-a == -1
//   bit  21      neg_b              imm-scale-b == -1
//   bit  22      trans_a            imm-trans-a == 1 (A is M-major in smem)
//   bit  23      trans_b            imm-trans-b == 1 (B is N-major in smem)
//   bit  24      satfinite          integer .satfinite
//   bits 25..26  sp_sel             sparsity selector (sp-sel)
//   bits 27..28  mode               0 exact, 1 fast, 2 check
//   bit  29      trap               check mode: __trap() after reporting a violation
//   bits 30..31  reserved, must be 0
//
// ============================================================================================
// Modes
//   exact: every thread gathers the full A rows (r, r+8) and the B columns of its D elements and
//     evaluates numerics::dot_float / dot_int / dot_b1_and_popc with wgmma_params (Hopper
//     T-FDPA). RS A rows are gathered from the quad lanes (4g..4g+3) with __shfl_sync; sparse
//     metadata is gathered with __shfl_sync from the lane given by spec::sp_meta_loc and only
//     the K/2 selected products enter one T-FDPA block. Warps never talk to each other.
//   fast: SM80 mma.sync per 16-row warp slice and 8-column n-tile, decomposition below.
//   check: exact numerics, but the result goes into the per-thread pending table and the
//     accumulator registers receive poison (f32 0x7FC0DEAD, f16x2 0x7E017E01, s32 0xDEADBEEF).
//
// fast-mode decomposition (per warp w: rows 16w..16w+15, per n-tile j: columns 8j..8j+7;
// the m16n8 C/D fragment equals wgmma D registers {4j..4j+3} (f32/s32) or {2j, 2j+1} (f16)):
//   f16  -> mma.m16n8k16.f32.f16.f16.f32 or .f16.f16.f16.f16, S = K/16 sub-mmas
//   bf16 -> mma.m16n8k16.f32.bf16.bf16.f32,                  S = K/16
//   tf32 -> mma.m16n8k8.f32.tf32.tf32.f32 (inputs & 0xFFFFE000), S = K/8
//   e4m3/e5m2 -> exact conversion to f16, then the f16 mma above, S = K/16. Sub-mma s, mma
//     column c maps to wgmma k = 16s + 4*(c/2) + (c%2) for c < 8 and 16s + 4*((c-8)/2) + 2 +
//     (c%2) for c >= 8 (this lets the four bytes of one wgmma A register feed one f16x2 pair
//     each of the two mma halves without shuffles).
//   s8/u8 -> mma.m16n8k32.s32.{s8|u8}.{s8|u8}.s32, S = K/32 (dense 1, sparse 2): chained mmas
//     from C = 0 without .satfinite (exact, |partial| < 2^22), then C is added in software with
//     64-bit saturation or 32-bit wrap, so the integer result equals the exact one. (SM80
//     mma .satfinite with C near the s32 limits did not match the single final saturation of
//     wgmma: e2e/direct fast mismatches 11-26, so C is never fed to the hardware mma.)
//   b1 -> mma.m16n8k256.s32.b1.b1.s32.and.popc, S = 1.
//   Sub-mmas are chained in s order (D of s is C of s+1); C of s = 0 is +0 when scale-d = 0.
//   Sparse A is expanded to a dense row (pruned positions hold +0) before the mmas; imm-scale-a
//   flips the stored values only. Element mapping inside a sub-mma follows the SM80 mma.sync
//   fragments (A: rows g / g+8, B: column g); RS A registers are used directly as the A
//   fragment (dense, non-fp8), which relies on the wgmma RS A layout equalling the mma.sync A
//   layout (tests/wgmma/test_direct.cu checks this on the host).
//
// ============================================================================================
// Check-mode table (per thread, in the translator's function-scope .local buffer):
//   bytes = WG_HDR_BYTES + n_acc * WG_ACC_SLOT_BYTES + n_a * WG_A_SLOT_BYTES
//         + WG_MAX_OPS * WG_OP_BYTES     (= 64 + 544*n_acc + 48*n_a + 3072)
//   Keep identical to rules_wgmma.py (table_bytes()).
// Violations are printed as
//   WGEMU-VIOLATION <kind> <detail> tid=<t> block=(x,y,z) acc=<id> x=0x.. y=0x..
// (at most two lines per thread) and recorded in TabHeader::violations, ::kinds_seen (bit
// V_* set) and ::last_kind. kinds: fence-missing, acc-modified-before-wait,
// acc-read-before-wait, rs-a-modified-before-wait, smem-modified-before-wait,
// sp-meta-invalid, acc-access-before-wait.
// ============================================================================================
#pragma once

#include <stdint.h>

#include "wgmma_emu/e4_device.cuh"  // E4 shadow shared memory mode (empty unless WGEMU_SMEM_GLOBAL)
#include "wgmma_emu/numerics.h"
#include "wgmma_emu/spec.h"
#include "wgmma_emu/types.h"

namespace wgemu {
namespace wg {

// ---------------------------------------------------------------------------------------------
// Variant encoding
// ---------------------------------------------------------------------------------------------
constexpr uint32_t VAR_N_SHIFT = 0;
constexpr uint32_t VAR_N_MASK = 0x3Fu;
constexpr uint32_t VAR_ATYPE_SHIFT = 6;
constexpr uint32_t VAR_BTYPE_SHIFT = 10;
constexpr uint32_t VAR_DTYPE_SHIFT = 14;
constexpr uint32_t VAR_TYPE_MASK = 0xFu;
constexpr uint32_t VAR_RS = 1u << 18;
constexpr uint32_t VAR_SPARSE = 1u << 19;
constexpr uint32_t VAR_NEG_A = 1u << 20;
constexpr uint32_t VAR_NEG_B = 1u << 21;
constexpr uint32_t VAR_TRANS_A = 1u << 22;
constexpr uint32_t VAR_TRANS_B = 1u << 23;
constexpr uint32_t VAR_SATFINITE = 1u << 24;
constexpr uint32_t VAR_SPSEL_SHIFT = 25;
constexpr uint32_t VAR_SPSEL_MASK = 0x3u;
constexpr uint32_t VAR_MODE_SHIFT = 27;
constexpr uint32_t VAR_MODE_MASK = 0x3u;
constexpr uint32_t VAR_TRAP = 1u << 29;
constexpr uint32_t VAR_RESERVED = 3u << 30;

constexpr uint32_t IDS_NO_A = 0xFFFFu;

enum class Mode : uint8_t { Exact = 0, Fast = 1, Check = 2 };

struct Variant {
  int n;
  EType at, bt, dt;
  bool rs, sparse, neg_a, neg_b, trans_a, trans_b, satfinite;
  int sp_sel;
  Mode mode;
  bool trap;
  uint32_t reserved;  // nonzero reserved bits (always invalid)
};

WGEMU_HD Variant decode_variant(uint32_t x) {
  Variant v;
  v.n = int(((x >> VAR_N_SHIFT) & VAR_N_MASK) + 1) * 8;
  v.at = EType((x >> VAR_ATYPE_SHIFT) & VAR_TYPE_MASK);
  v.bt = EType((x >> VAR_BTYPE_SHIFT) & VAR_TYPE_MASK);
  v.dt = EType((x >> VAR_DTYPE_SHIFT) & VAR_TYPE_MASK);
  v.rs = (x & VAR_RS) != 0;
  v.sparse = (x & VAR_SPARSE) != 0;
  v.neg_a = (x & VAR_NEG_A) != 0;
  v.neg_b = (x & VAR_NEG_B) != 0;
  v.trans_a = (x & VAR_TRANS_A) != 0;
  v.trans_b = (x & VAR_TRANS_B) != 0;
  v.satfinite = (x & VAR_SATFINITE) != 0;
  v.sp_sel = int((x >> VAR_SPSEL_SHIFT) & VAR_SPSEL_MASK);
  v.mode = Mode((x >> VAR_MODE_SHIFT) & VAR_MODE_MASK);
  v.trap = (x & VAR_TRAP) != 0;
  v.reserved = x & VAR_RESERVED;
  return v;
}

WGEMU_HD uint32_t encode_variant(const Variant& v) {
  uint32_t x = 0;
  x |= (uint32_t(v.n / 8 - 1) & VAR_N_MASK) << VAR_N_SHIFT;
  x |= (uint32_t(v.at) & VAR_TYPE_MASK) << VAR_ATYPE_SHIFT;
  x |= (uint32_t(v.bt) & VAR_TYPE_MASK) << VAR_BTYPE_SHIFT;
  x |= (uint32_t(v.dt) & VAR_TYPE_MASK) << VAR_DTYPE_SHIFT;
  if (v.rs) x |= VAR_RS;
  if (v.sparse) x |= VAR_SPARSE;
  if (v.neg_a) x |= VAR_NEG_A;
  if (v.neg_b) x |= VAR_NEG_B;
  if (v.trans_a) x |= VAR_TRANS_A;
  if (v.trans_b) x |= VAR_TRANS_B;
  if (v.satfinite) x |= VAR_SATFINITE;
  x |= (uint32_t(v.sp_sel) & VAR_SPSEL_MASK) << VAR_SPSEL_SHIFT;
  x |= (uint32_t(v.mode) & VAR_MODE_MASK) << VAR_MODE_SHIFT;
  if (v.trap) x |= VAR_TRAP;
  return x;
}

WGEMU_HD Variant make_variant(int n, EType at, EType bt, EType dt) {
  Variant v;
  v.n = n;
  v.at = at;
  v.bt = bt;
  v.dt = dt;
  v.rs = v.sparse = v.neg_a = v.neg_b = v.trans_a = v.trans_b = v.satfinite = false;
  v.sp_sel = 0;
  v.mode = Mode::Exact;
  v.trap = false;
  v.reserved = 0;
  return v;
}

// Logical K of one wgmma (PTX 9.5 shape table).
WGEMU_HD int wg_k(EType at, bool sparse) {
  switch (at) {
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
      return 256;
    default:
      return 0;
  }
}

WGEMU_HD bool wg_is_int(EType t) { return t == EType::S8 || t == EType::U8 || t == EType::B1; }

// Valid N: floating point 8..256 step 8; integer and b1 8, 16, 24, 32 and 48..256 step 16
// (the PTX 9.5 shape table lists n240/n256 for integer even where the syntax block stops at
// n224; the superset is accepted, PLAN section 7).
WGEMU_HD bool wg_valid_n(EType at, int n) {
  if (n < 8 || n > 256 || (n % 8) != 0) return false;
  if (wg_is_int(at)) return n <= 32 || (n % 16) == 0;
  return true;
}

// Accumulator registers in the PTX d vector.
WGEMU_HD int d_regs(const Variant& v) { return v.dt == EType::F16 ? v.n / 4 : v.n / 2; }

// nullptr when the variant is a legal PTX 9.5 wgmma.mma_async{.sp}; otherwise a short reason.
WGEMU_HD const char* variant_error(const Variant& v) {
  if (v.reserved) return "reserved-bits";
  if (uint32_t(v.mode) > 2) return "mode";
  if (!wg_valid_n(v.at, v.n)) return "n";
  const bool f16d = v.dt == EType::F16, f32d = v.dt == EType::F32, s32d = v.dt == EType::S32;
  switch (v.at) {
    case EType::F16:
      if (v.bt != EType::F16 || !(f16d || f32d)) return "types";
      break;
    case EType::BF16:
      if (v.bt != EType::BF16 || !f32d) return "types";
      break;
    case EType::TF32:
      if (v.bt != EType::TF32 || !f32d) return "types";
      break;
    case EType::E4M3:
    case EType::E5M2:
      if (!etype_is_fp8(v.bt) || !(f16d || f32d)) return "types";
      break;
    case EType::S8:
    case EType::U8:
      if (!etype_is_int8(v.bt) || !s32d) return "types";
      break;
    case EType::B1:
      if (v.bt != EType::B1 || !s32d) return "types";
      if (v.sparse) return "b1-sparse";
      break;
    default:
      return "types";
  }
  const bool trans_ok = v.at == EType::F16 || v.at == EType::BF16;
  if (v.trans_a && (!trans_ok || v.rs)) return "trans-a";
  if (v.trans_b && !trans_ok) return "trans-b";
  if ((v.neg_a || v.neg_b) && !etype_is_float_input(v.at)) return "imm-scale";
  if (v.satfinite && !etype_is_int8(v.at)) return "satfinite";
  if (!v.sparse) {
    if (v.sp_sel != 0) return "sp-sel";
  } else if (v.at == EType::F16 || v.at == EType::BF16 || v.at == EType::TF32) {
    if (v.sp_sel > 1) return "sp-sel";
  } else if (v.sp_sel != 0) {
    return "sp-sel";
  }
  return nullptr;
}

// Sign bit of one input element and of one packed 32-bit input register.
WGEMU_HD uint32_t elem_sign(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 0x8000u;
    case EType::TF32:
      return 0x80000000u;
    case EType::E4M3:
    case EType::E5M2:
      return 0x80u;
    default:
      return 0;
  }
}
WGEMU_HD uint32_t reg_sign(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 0x80008000u;
    case EType::TF32:
      return 0x80000000u;
    case EType::E4M3:
    case EType::E5M2:
      return 0x80808080u;
    default:
      return 0;
  }
}

// Bits that one element occupies when loaded from smem (b1 is handled as 32-bit words).
WGEMU_HD int load_bits(EType t) { return t == EType::B1 ? 32 : etype_bits(t); }

// Exact e4m3/e5m2 -> f16 conversion (all e4m3/e5m2 values are representable in f16).
WGEMU_HD uint32_t fp8_to_f16(uint32_t x, EType t) {
  x &= 0xFFu;
  if (t == EType::E5M2) return x << 8;
  const uint32_t s = (x & 0x80u) << 8;
  const uint32_t e = (x >> 3) & 0xFu, m = x & 7u;
  if (e == 15u && m == 7u) return s | 0x7E00u;  // NaN
  if (e == 0u) {
    if (m == 0u) return s;
    const uint32_t p = m >= 4u ? 2u : (m >= 2u ? 1u : 0u);  // floor(log2(m))
    return s | ((p + 6u) << 10) | ((m - (1u << p)) << (10u - p));
  }
  return s | ((e + 8u) << 10) | (m << 7);
}
WGEMU_HD uint32_t fp8x2_to_f16x2(uint32_t x16, EType t) {
  return fp8_to_f16(x16 & 0xFFu, t) | (fp8_to_f16((x16 >> 8) & 0xFFu, t) << 16);
}

// D element access. f16 D: register i holds elements 2i (low half) and 2i+1 (high half).
WGEMU_HD uint32_t d_get(const uint32_t* r, bool f16d, int i) {
  return f16d ? ((r[i >> 1] >> ((i & 1) * 16)) & 0xFFFFu) : r[i];
}
WGEMU_HD void d_set(uint32_t* r, bool f16d, int i, uint32_t v) {
  if (f16d) {
    const int s = (i & 1) * 16;
    r[i >> 1] = (r[i >> 1] & ~(0xFFFFu << s)) | ((v & 0xFFFFu) << s);
  } else {
    r[i] = v;
  }
}

WGEMU_HD uint32_t poison_of(EType dt) {
  return dt == EType::F16 ? 0x7E017E01u : (dt == EType::S32 ? 0xDEADBEEFu : 0x7FC0DEADu);
}

// ---------------------------------------------------------------------------------------------
// Check-mode table layout
// ---------------------------------------------------------------------------------------------
constexpr uint32_t WG_MAGIC = 0x57474D31u;  // "WGM1"
constexpr uint32_t WG_MAX_OPS = 64;
constexpr uint32_t WG_MAX_DREGS = 128;
constexpr uint32_t WG_FLAG_TRAP = 1u;

enum : uint32_t { ST_IDLE = 0, ST_PENDING = 1, ST_COMPLETE = 2 };

// Violation kinds (bit index in TabHeader::kinds_seen).
enum : uint32_t {
  V_FENCE_MISSING = 0,
  V_ACC_MODIFIED = 1,
  V_ACC_READ = 2,
  V_RS_A_MODIFIED = 3,
  V_SMEM_MODIFIED = 4,
  V_SP_META = 5,
  V_ACC_ACCESS = 6,
  V_DESC_UNDEFINED = 7,
  V_NUM_KINDS = 8
};

WGEMU_HD const char* violation_name(uint32_t k) {
  switch (k) {
    case V_FENCE_MISSING:
      return "fence-missing";
    case V_ACC_MODIFIED:
      return "acc-modified-before-wait";
    case V_ACC_READ:
      return "acc-read-before-wait";
    case V_RS_A_MODIFIED:
      return "rs-a-modified-before-wait";
    case V_SMEM_MODIFIED:
      return "smem-modified-before-wait";
    case V_SP_META:
      return "sp-meta-invalid";
    case V_ACC_ACCESS:
      return "acc-access-before-wait";
    case V_DESC_UNDEFINED:
      return "desc-undefined";
    default:
      return "unknown";
  }
}

struct TabHeader {
  uint32_t magic;
  uint32_t n_acc;
  uint32_t n_a;
  uint32_t max_ops;
  uint32_t flags;
  uint32_t cur_group;    // number of wgmma.commit_group executed so far
  uint32_t fence_count;  // number of wgmma.fence executed so far
  uint32_t violations;
  uint32_t printed;
  uint32_t issued;      // wgmma.mma_async issued in check mode
  uint32_t kinds_seen;  // bit k set when a violation of kind k (V_*) was reported
  uint32_t last_kind;   // kind of the most recent violation
  uint32_t pad[4];
};
struct AccSlot {
  uint32_t state;
  uint32_t nregs;
  uint32_t variant;  // variant of the most recent op on this accumulator
  uint32_t n_pending;
  uint32_t fence_at_issue;
  uint32_t fence_at_fetch;
  uint32_t has_handback;  // values[] equals what fetch handed back to the registers
  uint32_t dtype;
  uint32_t values[WG_MAX_DREGS];
};
struct ASlot {
  uint32_t state;
  uint32_t n_pending;
  uint32_t fence_at_issue;
  uint32_t used;
  uint32_t a[4];
  uint32_t pad[4];
};
struct OpRec {
  uint32_t valid;
  uint32_t acc_id;
  uint32_t a_id;
  uint32_t seq;
  uint32_t variant;
  uint32_t pad;
  uint64_t adesc;
  uint64_t bdesc;
  uint64_t hash;
};

constexpr uint32_t WG_HDR_BYTES = 64;
constexpr uint32_t WG_ACC_SLOT_BYTES = 544;
constexpr uint32_t WG_A_SLOT_BYTES = 48;
constexpr uint32_t WG_OP_BYTES = 48;
static_assert(sizeof(TabHeader) == WG_HDR_BYTES, "table header layout");
static_assert(sizeof(AccSlot) == WG_ACC_SLOT_BYTES, "acc slot layout");
static_assert(sizeof(ASlot) == WG_A_SLOT_BYTES, "a slot layout");
static_assert(sizeof(OpRec) == WG_OP_BYTES, "op record layout");

WGEMU_HD uint32_t table_bytes(uint32_t n_acc, uint32_t n_a) {
  return WG_HDR_BYTES + n_acc * WG_ACC_SLOT_BYTES + n_a * WG_A_SLOT_BYTES +
         WG_MAX_OPS * WG_OP_BYTES;
}

}  // namespace wg
}  // namespace wgemu

// =============================================================================================
// Device implementation
// =============================================================================================
#if defined(__CUDACC__)

namespace wgemu {
namespace wg {

constexpr unsigned FULL = 0xFFFFFFFFu;

WGEMU_DEV uint32_t linear_tid() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

// Hard failure inside the emulation itself (never a program-level violation).
inline __device__ __noinline__ void fatal(const char* what, uint32_t x) {
  printf("WGEMU-ERROR wgmma %s 0x%08x tid=%u block=(%u,%u,%u)\n", what, x, linear_tid(), blockIdx.x,
         blockIdx.y, blockIdx.z);
  __trap();
}

#ifdef WGEMU_SMEM_GLOBAL
WGEMU_DEV uint32_t lds(uint32_t addr, int bits) { return e4::lds(addr, bits); }
#else
WGEMU_DEV uint32_t lds(uint32_t addr, int bits) {
  uint32_t v;
  if (bits == 8) {
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  } else if (bits == 16) {
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  } else {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  }
  return v;
}
#endif

WGEMU_DEV uint32_t elem_addr(const spec::Desc& d, bool mn_major, EType t, int mn, int k) {
  return spec::smem_addr(d, mn_major, load_bits(t), mn, k);
}
WGEMU_DEV uint32_t load_elem(const spec::Desc& d, bool mn_major, EType t, int mn, int k) {
  return lds(elem_addr(d, mn_major, t, mn, k), load_bits(t));
}

// Values one thread needs from A: rows r and r+8 of its warp slice.
struct ARows {
  uint32_t v[2][32];  // dense: logical K values (b1: 8 words); sparse: K/2 compressed values
  uint8_t k[2][32];   // sparse: logical k of each compressed value
  int n;              // values per row
  bool bad_meta;      // sparse metadata outside the PTX-defined value set
};

// Row (0..63) of A row half h for thread t.
WGEMU_DEV int row_of(uint32_t t, int h) { return int(16 * (t >> 5) + ((t & 31) >> 2) + 8 * h); }

WGEMU_DEV void gather_a(const Variant& v, uint64_t adesc, const uint32_t a[4], uint32_t meta,
                        uint32_t t, ARows& R) {
  const int lane = int(t & 31);
  const int m0 = row_of(t, 0);
  const int K = wg_k(v.at, v.sparse);
  const int na = v.at == EType::B1 ? 8 : (v.sparse ? K / 2 : K);
  R.n = na;
  R.bad_meta = false;
  if (v.rs) {
    const int bits = load_bits(v.at);
    const int per = 32 / bits;
    const uint32_t mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u);
#pragma unroll 1
    for (int qs = 0; qs < 4; ++qs) {
      const int src = (lane & ~3) | qs;
      const int ts = int(t & ~31u) | src;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t x = __shfl_sync(FULL, a[j], src);
#pragma unroll 1
        for (int e = 0; e < per; ++e) {
          int m = -1, k = -1;
          if (v.sparse) {
            spec::sp_a_frag_coord(v.at, ts, j, e, &m, &k);
          } else {
            spec::a_frag_coord(v.at, ts, j, e, &m, &k);
          }
          if (v.at == EType::B1) {
            if (k & 31) fatal("spec-a-frag-b1", uint32_t(k));
            k >>= 5;
          }
          const int h = (m == m0) ? 0 : (m == m0 + 8 ? 1 : -1);
          if (h < 0 || k < 0 || k >= na) fatal("spec-a-frag", uint32_t(m << 16 | (k & 0xFFFF)));
          R.v[h][k] = (x >> (e * bits)) & mask;
        }
      }
    }
  } else {
    const spec::Desc d = spec::decode_desc(adesc);
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
#pragma unroll 1
      for (int kk = 0; kk < na; ++kk) R.v[h][kk] = load_elem(d, v.trans_a, v.at, m0 + 8 * h, kk);
    }
  }
  if (v.neg_a) {
    const uint32_t s = elem_sign(v.at);
    for (int h = 0; h < 2; ++h)
      for (int kk = 0; kk < na; ++kk) R.v[h][kk] ^= s;
  }
  if (v.sparse) {
    const bool tf = v.at == EType::TF32;
    const int nch = tf ? K / 2 : K / 4;
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
#pragma unroll 1
      for (int c = 0; c < nch; ++c) {
        int ln = -1, bit = -1;
        spec::sp_meta_loc(v.at, v.sp_sel, m0 + 8 * h, c, &ln, &bit);
        if (ln < 0 || ln > 31 || bit < 0 || bit > 28) fatal("spec-meta-loc", uint32_t(c));
        const uint32_t mv = __shfl_sync(FULL, meta, ln);
        const uint32_t f = (mv >> bit) & 0xFu;
        int i0 = -1, i1 = -1;
        spec::sp_meta_decode(v.at, f, &i0, &i1);
        if (tf) {
          if (i0 < 0 || i0 > 1) fatal("spec-meta-decode", f);
          if (f != 0x4u && f != 0xEu) R.bad_meta = true;
          R.k[h][c] = uint8_t(2 * c + i0);
        } else {
          if (i0 < 0 || i0 > 3 || i1 < 0 || i1 > 3) fatal("spec-meta-decode", f);
          if (i0 == i1) R.bad_meta = true;
          R.k[h][2 * c] = uint8_t(4 * c + i0);
          R.k[h][2 * c + 1] = uint8_t(4 * c + i1);
        }
      }
    }
  }
}

// Logical-K values of B column n (b1: 8 words).
WGEMU_DEV int b_count(const Variant& v) { return v.at == EType::B1 ? 8 : wg_k(v.at, v.sparse); }

WGEMU_DEV uint32_t dot_any(const Variant& v, const uint32_t* a, const uint32_t* b, int n,
                           uint32_t c, const FdpaParams& p) {
  if (v.at == EType::B1) return uint32_t(dot_b1_and_popc(a, b, n, int32_t(c)));
  if (etype_is_int8(v.at)) {
    return uint32_t(
        dot_int(a, b, n, v.at == EType::S8, v.bt == EType::S8, int32_t(c), v.satfinite));
  }
  return dot_float(a, b, n, v.at, v.bt, c, v.dt == EType::F16, p);
}

// D element pair (i, i+2) of n-tile j / column parity e share one B column. Returns the column
// and the two element rows (verified against the thread's rows).
WGEMU_DEV int d_pair(uint32_t t, int j, int e, int* i_out, int* h_out) {
  const int m0 = row_of(t, 0);
  int col = -1;
  for (int s = 0; s < 2; ++s) {
    const int i = 4 * j + 2 * s + e;
    int m = -1, n = -1;
    spec::d_frag_coord(int(t), i, &m, &n);
    const int h = (m == m0) ? 0 : (m == m0 + 8 ? 1 : -1);
    if (h < 0 || (s == 1 && n != col)) fatal("spec-d-frag", uint32_t(i));
    col = n;
    i_out[s] = i;
    h_out[s] = h;
  }
  return col;
}

WGEMU_DEV void exact_compute(const Variant& v, const ARows& R, uint64_t bdesc, uint32_t t,
                             const uint32_t* cin, bool use_c, uint32_t* dout) {
  const bool f16d = v.dt == EType::F16;
  const spec::Desc dB = spec::decode_desc(bdesc);
  const int nb = b_count(v);
  const uint32_t bneg = v.neg_b ? elem_sign(v.bt) : 0u;
  FdpaParams p = {};
  if (etype_is_float_input(v.at)) p = wgmma_params(v.at, v.bt, f16d);
  uint32_t col[64];
  uint32_t bsel[32];
#pragma unroll 1
  for (int j = 0; j < v.n / 8; ++j) {
#pragma unroll 1
    for (int e = 0; e < 2; ++e) {
      int ii[2], hh[2];
      const int n = d_pair(t, j, e, ii, hh);
#pragma unroll 1
      for (int k = 0; k < nb; ++k) col[k] = load_elem(dB, v.trans_b, v.bt, n, k) ^ bneg;
#pragma unroll 1
      for (int s = 0; s < 2; ++s) {
        const int h = hh[s];
        const uint32_t c = use_c ? d_get(cin, f16d, ii[s]) : 0u;
        uint32_t r;
        if (v.sparse) {
          for (int kc = 0; kc < R.n; ++kc) bsel[kc] = col[R.k[h][kc]];
          r = dot_any(v, R.v[h], bsel, R.n, c, p);
        } else {
          r = dot_any(v, R.v[h], col, R.n, c, p);
        }
        d_set(dout, f16d, ii[s], r);
      }
    }
  }
}

// Digest of every smem element the thread reads for this op (A rows when SS, B columns).
WGEMU_DEV uint64_t smem_digest(const Variant& v, uint64_t adesc, uint64_t bdesc, uint32_t t) {
  uint64_t h = 0xCBF29CE484222325ull;
  auto mix = [&h](uint32_t addr, uint32_t val) {
    h ^= (uint64_t(addr) << 32) | val;
    h *= 0x100000001B3ull;
    h ^= h >> 29;
  };
  if (!v.rs) {
    const spec::Desc dA = spec::decode_desc(adesc);
    const int na = v.at == EType::B1 ? 8 : (v.sparse ? wg_k(v.at, true) / 2 : wg_k(v.at, false));
    for (int hh = 0; hh < 2; ++hh)
      for (int k = 0; k < na; ++k) {
        const uint32_t ad = elem_addr(dA, v.trans_a, v.at, row_of(t, hh), k);
        mix(ad, lds(ad, load_bits(v.at)));
      }
  }
  const spec::Desc dB = spec::decode_desc(bdesc);
  const int nb = b_count(v);
  for (int j = 0; j < v.n / 8; ++j)
    for (int e = 0; e < 2; ++e) {
      int ii[2], hh[2];
      const int n = d_pair(t, j, e, ii, hh);
      for (int k = 0; k < nb; ++k) {
        const uint32_t ad = elem_addr(dB, v.trans_b, v.bt, n, k);
        mix(ad, lds(ad, load_bits(v.bt)));
      }
    }
  return h;
}

// ---------------------------------------------------------------------------------------------
// fast mode: SM80 mma.sync wrappers
// ---------------------------------------------------------------------------------------------
WGEMU_DEV void mma_f16_f32(const uint32_t* a, const uint32_t* b, uint32_t* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
WGEMU_DEV void mma_f16_f16(const uint32_t* a, const uint32_t* b, uint32_t* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
      "{%0,%1};"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
WGEMU_DEV void mma_bf16_f32(const uint32_t* a, const uint32_t* b, uint32_t* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
WGEMU_DEV void mma_tf32_f32(const uint32_t* a, const uint32_t* b, uint32_t* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#define WGEMU_MMA_I8(NAME, SAT, AT, BT)                                               \
  WGEMU_DEV void NAME(const uint32_t* a, const uint32_t* b, uint32_t* c) {            \
    asm volatile("mma.sync.aligned.m16n8k32.row.col" SAT ".s32." AT "." BT            \
                 ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"         \
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])                     \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])); \
  }
WGEMU_MMA_I8(mma_s8s8, "", "s8", "s8")
WGEMU_MMA_I8(mma_s8u8, "", "s8", "u8")
WGEMU_MMA_I8(mma_u8s8, "", "u8", "s8")
WGEMU_MMA_I8(mma_u8u8, "", "u8", "u8")
WGEMU_MMA_I8(mma_s8s8_sat, ".satfinite", "s8", "s8")
WGEMU_MMA_I8(mma_s8u8_sat, ".satfinite", "s8", "u8")
WGEMU_MMA_I8(mma_u8s8_sat, ".satfinite", "u8", "s8")
WGEMU_MMA_I8(mma_u8u8_sat, ".satfinite", "u8", "u8")
#undef WGEMU_MMA_I8
WGEMU_DEV void mma_b1(const uint32_t* a, const uint32_t* b, uint32_t* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

WGEMU_DEV void mma_i8(const Variant& v, bool sat, const uint32_t* a, const uint32_t* b,
                      uint32_t* c) {
  const bool as = v.at == EType::S8, bs = v.bt == EType::S8;
  if (sat) {
    if (as && bs)
      mma_s8s8_sat(a, b, c);
    else if (as)
      mma_s8u8_sat(a, b, c);
    else if (bs)
      mma_u8s8_sat(a, b, c);
    else
      mma_u8u8_sat(a, b, c);
  } else {
    if (as && bs)
      mma_s8s8(a, b, c);
    else if (as)
      mma_s8u8(a, b, c);
    else if (bs)
      mma_u8s8(a, b, c);
    else
      mma_u8u8(a, b, c);
  }
}

// Number of sub-mmas per n-tile.
WGEMU_HD int fast_subs(const Variant& v) {
  const int K = wg_k(v.at, v.sparse);
  switch (v.at) {
    case EType::F16:
    case EType::BF16:
    case EType::E4M3:
    case EType::E5M2:
      return K / 16;
    case EType::TF32:
      return K / 8;
    case EType::S8:
    case EType::U8:
      return K / 32;
    default:
      return 1;
  }
}

// Build the m16n8kX operand fragment of sub-mma s from an element accessor get(h, k) (A: h is
// the row half g / g+8, k the logical K; B: h unused). For A, R has 4 registers; for B, 2.
// The register/element placement is the SM80 mma.sync fragment layout; fp8 uses the k mapping
// documented in the header comment.
template <class Get>
WGEMU_DEV void fast_frag_a(const Variant& v, int s, int q, Get get, uint32_t* R) {
  switch (v.at) {
    case EType::F16:
    case EType::BF16:
      for (int r = 0; r < 4; ++r) {
        const int h = r & 1, k = 16 * s + 8 * (r >> 1) + 2 * q;
        R[r] = (get(h, k) & 0xFFFFu) | ((get(h, k + 1) & 0xFFFFu) << 16);
      }
      break;
    case EType::E4M3:
    case EType::E5M2:
      for (int r = 0; r < 4; ++r) {
        const int h = r & 1, k = 16 * s + 4 * q + 2 * (r >> 1);
        R[r] = fp8_to_f16(get(h, k), v.at) | (fp8_to_f16(get(h, k + 1), v.at) << 16);
      }
      break;
    case EType::TF32:
      for (int r = 0; r < 4; ++r) {
        const int h = r & 1, k = 8 * s + q + 4 * (r >> 1);
        R[r] = get(h, k) & 0xFFFFE000u;
      }
      break;
    case EType::S8:
    case EType::U8:
      for (int r = 0; r < 4; ++r) {
        const int h = r & 1, k = 32 * s + 16 * (r >> 1) + 4 * q;
        R[r] = (get(h, k) & 0xFFu) | ((get(h, k + 1) & 0xFFu) << 8) |
               ((get(h, k + 2) & 0xFFu) << 16) | ((get(h, k + 3) & 0xFFu) << 24);
      }
      break;
    case EType::B1:
      for (int r = 0; r < 4; ++r) R[r] = get(r & 1, 4 * (r >> 1) + q);  // word index
      break;
    default:
      break;
  }
}
template <class Get>
WGEMU_DEV void fast_frag_b(const Variant& v, int s, int q, Get get, uint32_t* R) {
  switch (v.bt) {
    case EType::F16:
    case EType::BF16:
      for (int r = 0; r < 2; ++r) {
        const int k = 16 * s + 8 * r + 2 * q;
        R[r] = (get(k) & 0xFFFFu) | ((get(k + 1) & 0xFFFFu) << 16);
      }
      break;
    case EType::E4M3:
    case EType::E5M2:
      for (int r = 0; r < 2; ++r) {
        const int k = 16 * s + 4 * q + 2 * r;
        R[r] = fp8_to_f16(get(k), v.bt) | (fp8_to_f16(get(k + 1), v.bt) << 16);
      }
      break;
    case EType::TF32:
      for (int r = 0; r < 2; ++r) R[r] = get(8 * s + q + 4 * r) & 0xFFFFE000u;
      break;
    case EType::S8:
    case EType::U8:
      for (int r = 0; r < 2; ++r) {
        const int k = 32 * s + 16 * r + 4 * q;
        R[r] = (get(k) & 0xFFu) | ((get(k + 1) & 0xFFu) << 8) | ((get(k + 2) & 0xFFu) << 16) |
               ((get(k + 3) & 0xFFu) << 24);
      }
      break;
    case EType::B1:
      for (int r = 0; r < 2; ++r) R[r] = get(4 * r + q);  // word index
      break;
    default:
      break;
  }
}

WGEMU_DEV void fast_compute(const Variant& v, uint64_t adesc, const uint32_t a_in[4], uint32_t meta,
                            uint64_t bdesc, uint32_t t, const uint32_t* cin, bool use_c,
                            uint32_t* dout) {
  const int lane = int(t & 31), g = lane >> 2, q = lane & 3;
  const int m0 = row_of(t, 0);
  const int S = fast_subs(v);
  const bool f16d = v.dt == EType::F16;
  uint32_t afr[4][4];
  if (!v.sparse && v.rs) {
    uint32_t a[4];
    const uint32_t neg = v.neg_a ? reg_sign(v.at) : 0u;
    for (int r = 0; r < 4; ++r) a[r] = a_in[r] ^ neg;
    if (etype_is_fp8(v.at)) {
      for (int s = 0; s < 2; ++s) {
        afr[s][0] = fp8x2_to_f16x2(a[2 * s] & 0xFFFFu, v.at);
        afr[s][1] = fp8x2_to_f16x2(a[2 * s + 1] & 0xFFFFu, v.at);
        afr[s][2] = fp8x2_to_f16x2(a[2 * s] >> 16, v.at);
        afr[s][3] = fp8x2_to_f16x2(a[2 * s + 1] >> 16, v.at);
      }
    } else {
      const uint32_t keep = v.at == EType::TF32 ? 0xFFFFE000u : 0xFFFFFFFFu;
      for (int r = 0; r < 4; ++r) afr[0][r] = a[r] & keep;
    }
  } else if (!v.sparse) {
    const spec::Desc dA = spec::decode_desc(adesc);
    const uint32_t neg = v.neg_a ? elem_sign(v.at) : 0u;
    auto get = [&](int h, int k) -> uint32_t {
      return load_elem(dA, v.trans_a, v.at, m0 + 8 * h, k) ^ neg;
    };
    for (int s = 0; s < S; ++s) fast_frag_a(v, s, q, get, afr[s]);
  } else {
    ARows R;
    gather_a(v, adesc, a_in, meta, t, R);  // neg_a already applied to the stored values
    uint32_t dense[2][64];
    const int K = wg_k(v.at, true);
    for (int h = 0; h < 2; ++h) {
      for (int k = 0; k < K; ++k) dense[h][k] = 0u;
      for (int kc = 0; kc < R.n; ++kc) dense[h][R.k[h][kc]] = R.v[h][kc];
    }
    auto get = [&](int h, int k) -> uint32_t { return dense[h][k]; };
    for (int s = 0; s < S; ++s) fast_frag_a(v, s, q, get, afr[s]);
  }

  const spec::Desc dB = spec::decode_desc(bdesc);
  const uint32_t bneg = v.neg_b ? elem_sign(v.bt) : 0u;
#pragma unroll 1
  for (int j = 0; j < v.n / 8; ++j) {
    const int n = 8 * j + g;
    auto getb = [&](int k) -> uint32_t { return load_elem(dB, v.trans_b, v.bt, n, k) ^ bneg; };
    uint32_t c[4];
    const int nc = f16d ? 2 : 4;
    for (int r = 0; r < nc; ++r) c[r] = use_c ? cin[nc * j + r] : 0u;
    if (etype_is_int8(v.at)) {
      uint32_t p[4] = {0u, 0u, 0u, 0u};
      for (int s = 0; s < S; ++s) {
        uint32_t bf[2];
        fast_frag_b(v, s, q, getb, bf);
        mma_i8(v, false, afr[s], bf, p);
      }
      for (int r = 0; r < 4; ++r) {
        if (v.satfinite) {
          int64_t x = int64_t(int32_t(c[r])) + int64_t(int32_t(p[r]));
          x = x > 2147483647ll ? 2147483647ll : (x < -2147483648ll ? -2147483648ll : x);
          c[r] = uint32_t(int32_t(x));
        } else {
          c[r] = c[r] + p[r];
        }
      }
    } else {
      for (int s = 0; s < S; ++s) {
        uint32_t bf[2];
        fast_frag_b(v, s, q, getb, bf);
        switch (v.at) {
          case EType::F16:
          case EType::E4M3:
          case EType::E5M2:
            if (f16d) {
              mma_f16_f16(afr[s], bf, c);
            } else {
              mma_f16_f32(afr[s], bf, c);
            }
            break;
          case EType::BF16:
            mma_bf16_f32(afr[s], bf, c);
            break;
          case EType::TF32:
            mma_tf32_f32(afr[s], bf, c);
            break;
          case EType::S8:
          case EType::U8:
            mma_i8(v, v.satfinite, afr[s], bf, c);
            break;
          case EType::B1:
            mma_b1(afr[s], bf, c);
            break;
          default:
            fatal("fast-type", uint32_t(v.at));
        }
      }
    }
    for (int r = 0; r < nc; ++r) dout[nc * j + r] = c[r];
  }
}

// ---------------------------------------------------------------------------------------------
// check mode
// ---------------------------------------------------------------------------------------------
struct Table {
  TabHeader* h;
  AccSlot* acc;
  ASlot* as;
  OpRec* ops;
};

WGEMU_DEV Table table_of(uint64_t pend) {
  Table tb;
  char* p = reinterpret_cast<char*>(pend);
  tb.h = reinterpret_cast<TabHeader*>(p);
  if (pend == 0 || tb.h->magic != WG_MAGIC) fatal("check-table-uninitialised", uint32_t(pend));
  tb.acc = reinterpret_cast<AccSlot*>(p + WG_HDR_BYTES);
  tb.as = reinterpret_cast<ASlot*>(p + WG_HDR_BYTES + tb.h->n_acc * WG_ACC_SLOT_BYTES);
  tb.ops = reinterpret_cast<OpRec*>(p + WG_HDR_BYTES + tb.h->n_acc * WG_ACC_SLOT_BYTES +
                                    tb.h->n_a * WG_A_SLOT_BYTES);
  return tb;
}

inline __device__ __noinline__ void report(TabHeader* h, bool trap, uint32_t kind,
                                           const char* detail, int acc, uint32_t x, uint32_t y) {
  h->violations++;
  h->kinds_seen |= 1u << (kind & 31u);
  h->last_kind = kind;
  if (h->printed < 2u) {
    h->printed++;
    printf("WGEMU-VIOLATION %s %s tid=%u block=(%u,%u,%u) acc=%d x=0x%08x y=0x%08x\n",
           violation_name(kind), detail, linear_tid(), blockIdx.x, blockIdx.y, blockIdx.z, acc, x,
           y);
  }
  if (trap || (h->flags & WG_FLAG_TRAP)) __trap();
}

WGEMU_DEV int first_diff(const uint32_t* a, const uint32_t* b, int n) {
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return -1;
}

WGEMU_DEV void check_init(uint64_t pend, uint32_t n_acc, uint32_t n_a, uint32_t flags) {
  if (pend == 0) fatal("check-init-null", 0u);
  if (n_acc > 0xFFFFu || n_a > 0xFFFFu) fatal("check-init-ids", n_acc);
  uint32_t* p = reinterpret_cast<uint32_t*>(pend);
  const uint32_t words = table_bytes(n_acc, n_a) / 4u;
  for (uint32_t i = 0; i < words; ++i) p[i] = 0u;
  TabHeader* h = reinterpret_cast<TabHeader*>(pend);
  h->magic = WG_MAGIC;
  h->n_acc = n_acc;
  h->n_a = n_a;
  h->max_ops = WG_MAX_OPS;
  h->flags = flags;
}

WGEMU_DEV void check_issue(const Variant& v, uint32_t variant, uint64_t adesc, const uint32_t a[4],
                           uint64_t bdesc, uint32_t meta, bool use_c, uint32_t ids, uint64_t pend,
                           uint32_t t, uint32_t* dp, const uint32_t* cur, int nregs) {
  Table tb = table_of(pend);
  TabHeader* H = tb.h;
  const uint32_t acc_id = ids & 0xFFFFu, a_id = ids >> 16;
  if (acc_id >= H->n_acc) fatal("check-acc-id", ids);
  if (v.rs && a_id >= H->n_a) fatal("check-a-id", ids);
  if (!v.rs && a_id != IDS_NO_A) fatal("check-a-id-ss", ids);
  AccSlot& S = tb.acc[acc_id];
  const uint32_t poison = poison_of(v.dt);
  const int ia = int(acc_id);
  H->issued++;

  if (S.state == ST_IDLE) {
    if (H->fence_count == 0u) {
      report(H, v.trap, V_FENCE_MISSING, "no-wgmma.fence-before-first-mma_async", ia, variant, 0u);
    } else if (S.has_handback && S.fence_at_fetch == H->fence_count && S.nregs == uint32_t(nregs)) {
      const int i = first_diff(cur, S.values, nregs);
      if (i >= 0) {
        report(H, v.trap, V_FENCE_MISSING, "accumulator-written-after-wait-without-fence", ia,
               cur[i], S.values[i]);
      }
    }
    if (use_c) {
      for (int i = 0; i < nregs; ++i) {
        if (cur[i] == poison) {
          report(H, v.trap, V_ACC_READ, "C-operand-holds-pending-poison", ia, cur[i], uint32_t(i));
          break;
        }
      }
    }
  } else {
    if (S.nregs != uint32_t(nregs) || S.dtype != uint32_t(v.dt)) fatal("check-acc-shape", ids);
    const Variant pv = decode_variant(S.variant);
    const bool same_shape =
        pv.n == v.n && wg_k(pv.at, pv.sparse) == wg_k(v.at, v.sparse) && pv.dt == v.dt;
    if (!same_shape && S.fence_at_issue == H->fence_count) {
      report(H, v.trap, V_FENCE_MISSING, "accumulator-chain-changes-shape-without-fence", ia,
             S.variant, variant);
    }
    for (int i = 0; i < nregs; ++i) {
      if (cur[i] != poison) {
        report(H, v.trap, V_ACC_MODIFIED, "register-written-while-pending", ia, cur[i],
               uint32_t(i));
        break;
      }
    }
  }
  if (v.rs) {
    ASlot& A = tb.as[a_id];
    if (A.state != ST_IDLE) {
      const int i = first_diff(a, A.a, 4);
      if (i >= 0) {
        report(H, v.trap, V_RS_A_MODIFIED, "A-register-changed-while-pending", ia, a[i], A.a[i]);
      }
    } else if (A.used && A.fence_at_issue == H->fence_count && H->fence_count != 0u) {
      const int i = first_diff(a, A.a, 4);
      if (i >= 0) {
        report(H, v.trap, V_FENCE_MISSING, "A-register-written-without-fence", ia, a[i], A.a[i]);
      }
    }
  }

  // Sparse B in K-major with 32B swizzle has no PTX-defined address for bytes 32..63 of a
  // row; the computation below uses spec.h's linear continuation (PLAN section 7).
  if (v.sparse && spec::sparse_b_layout_undefined(spec::decode_desc(bdesc), v.trans_b))
    report(H, v.trap, V_DESC_UNDEFINED, "sparse-B-K-major-32B-swizzle", ia, uint32_t(bdesc),
           uint32_t(bdesc >> 32));

  ARows R;
  gather_a(v, adesc, a, meta, t, R);
  if (R.bad_meta) report(H, v.trap, V_SP_META, "undefined-sparsity-metadata", ia, meta, 0u);

  uint32_t res[WG_MAX_DREGS];
  for (int i = 0; i < nregs; ++i) res[i] = 0u;
  const uint32_t* cin = (S.state == ST_IDLE) ? cur : S.values;
  exact_compute(v, R, bdesc, t, cin, use_c, res);

  uint32_t slot = WG_MAX_OPS;
  for (uint32_t i = 0; i < WG_MAX_OPS; ++i) {
    if (!tb.ops[i].valid) {
      slot = i;
      break;
    }
  }
  if (slot == WG_MAX_OPS) fatal("check-pending-overflow", WG_MAX_OPS);
  OpRec& op = tb.ops[slot];
  op.valid = 1u;
  op.acc_id = acc_id;
  op.a_id = v.rs ? a_id : IDS_NO_A;
  op.seq = H->cur_group;
  op.variant = variant;
  op.adesc = adesc;
  op.bdesc = bdesc;
  op.hash = smem_digest(v, adesc, bdesc, t);

  S.state = ST_PENDING;
  S.n_pending++;
  S.nregs = uint32_t(nregs);
  S.dtype = uint32_t(v.dt);
  S.variant = variant;
  S.fence_at_issue = H->fence_count;
  S.has_handback = 0u;
  for (int i = 0; i < nregs; ++i) S.values[i] = res[i];
  if (v.rs) {
    ASlot& A = tb.as[a_id];
    A.state = ST_PENDING;
    A.n_pending++;
    A.used = 1u;
    A.fence_at_issue = H->fence_count;
    for (int i = 0; i < 4; ++i) A.a[i] = a[i];
  }
  for (int i = 0; i < nregs; ++i) dp[i] = poison;
}

WGEMU_DEV void check_fence(uint64_t pend) { table_of(pend).h->fence_count++; }
WGEMU_DEV void check_commit(uint64_t pend) { table_of(pend).h->cur_group++; }

WGEMU_DEV void check_wait(uint64_t pend, uint32_t n) {
  Table tb = table_of(pend);
  TabHeader* H = tb.h;
  const uint32_t t = linear_tid() & 127u;
  for (uint32_t i = 0; i < WG_MAX_OPS; ++i) {
    OpRec& op = tb.ops[i];
    if (!op.valid) continue;
    // complete every op whose group seq < cur_group - n (uncommitted ops have seq == cur_group)
    if (uint64_t(op.seq) + uint64_t(n) >= uint64_t(H->cur_group)) continue;
    const Variant v = decode_variant(op.variant);
    const uint64_t hnow = smem_digest(v, op.adesc, op.bdesc, t);
    if (hnow != op.hash) {
      report(H, v.trap, V_SMEM_MODIFIED, "operand-smem-changed-between-issue-and-wait",
             int(op.acc_id), uint32_t(op.hash), uint32_t(hnow));
    }
    AccSlot& S = tb.acc[op.acc_id];
    if (S.n_pending == 0u) fatal("check-acc-count", op.acc_id);
    if (--S.n_pending == 0u) S.state = ST_COMPLETE;
    if (op.a_id != IDS_NO_A) {
      ASlot& A = tb.as[op.a_id];
      if (A.n_pending == 0u) fatal("check-a-count", op.a_id);
      if (--A.n_pending == 0u) A.state = ST_COMPLETE;
    }
    op.valid = 0u;
  }
}

WGEMU_DEV uint32_t check_fetch(uint64_t pend, uint32_t acc_id, uint64_t d_ptr) {
  Table tb = table_of(pend);
  TabHeader* H = tb.h;
  if (acc_id >= H->n_acc) fatal("fetch-acc-id", acc_id);
  AccSlot& S = tb.acc[acc_id];
  if (S.state == ST_IDLE) return 0u;
  uint32_t* dp = reinterpret_cast<uint32_t*>(d_ptr);
  const uint32_t poison = poison_of(EType(S.dtype));
  const int nregs = int(S.nregs);
  for (int i = 0; i < nregs; ++i) {
    if (dp[i] != poison) {
      report(H, false, V_ACC_MODIFIED, "register-written-while-pending", int(acc_id), dp[i],
             uint32_t(i));
      break;
    }
  }
  if (S.state == ST_PENDING) return 0u;
  for (int i = 0; i < nregs; ++i) dp[i] = S.values[i];
  S.state = ST_IDLE;
  S.has_handback = 1u;
  S.fence_at_fetch = H->fence_count;
  return 1u;
}

WGEMU_DEV void check_a_regs(uint64_t pend, uint32_t a_id, const uint32_t a[4]) {
  Table tb = table_of(pend);
  TabHeader* H = tb.h;
  if (a_id >= H->n_a) fatal("check-a-id", a_id);
  ASlot& A = tb.as[a_id];
  if (A.state == ST_IDLE) return;
  const int i = first_diff(a, A.a, 4);
  if (i >= 0) {
    report(H, false, V_RS_A_MODIFIED, "A-register-changed-while-pending", -1, a[i], A.a[i]);
  }
  if (A.state == ST_COMPLETE) A.state = ST_IDLE;
}

WGEMU_DEV void check_acc_access(uint64_t pend, uint32_t acc_id, uint32_t kind) {
  Table tb = table_of(pend);
  TabHeader* H = tb.h;
  if (acc_id >= H->n_acc) fatal("check-acc-id", acc_id);
  if ((kind & 1u) == 0u) return;  // guarded instruction whose guard is false
  if (tb.acc[acc_id].state != ST_IDLE) {
    report(H, false, V_ACC_ACCESS, "instruction-touches-pending-accumulator", int(acc_id), kind,
           tb.acc[acc_id].state);
  }
}

// One wgmma.mma_async{.sp} (all modes).
WGEMU_DEV void issue(uint64_t d_ptr, uint64_t adesc, uint32_t a0, uint32_t a1, uint32_t a2,
                     uint32_t a3, uint64_t bdesc, uint32_t meta, uint32_t scale_d, uint32_t variant,
                     uint32_t ids, uint64_t pend) {
  const Variant v = decode_variant(variant);
  if (const char* err = variant_error(v)) {
    printf("WGEMU-ERROR wgmma bad-variant %s 0x%08x\n", err, variant);
    __trap();
  }
  const uint32_t t = linear_tid() & 127u;
  const int nregs = d_regs(v);
  uint32_t* dp = reinterpret_cast<uint32_t*>(d_ptr);
  uint32_t cur[WG_MAX_DREGS];
  uint32_t res[WG_MAX_DREGS];
  for (int i = 0; i < nregs; ++i) {
    cur[i] = dp[i];
    res[i] = 0u;
  }
  const uint32_t a[4] = {a0, a1, a2, a3};
  const bool use_c = scale_d != 0u;
  switch (v.mode) {
    case Mode::Exact: {
      ARows R;
      gather_a(v, adesc, a, meta, t, R);
      exact_compute(v, R, bdesc, t, cur, use_c, res);
      for (int i = 0; i < nregs; ++i) dp[i] = res[i];
      break;
    }
    case Mode::Fast:
      fast_compute(v, adesc, a, meta, bdesc, t, cur, use_c, res);
      for (int i = 0; i < nregs; ++i) dp[i] = res[i];
      break;
    case Mode::Check:
      check_issue(v, variant, adesc, a, bdesc, meta, use_c, ids, pend, t, dp, cur, nregs);
      break;
    default:
      fatal("mode", variant);
  }
}

}  // namespace wg
}  // namespace wgemu

#endif  // __CUDACC__
