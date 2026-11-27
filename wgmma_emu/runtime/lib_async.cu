// Asynchronous-copy, TMA, mbarrier and tensor-map emulation library (PLAN section 10). Owner: A.
// Included by runtime/emu_lib.cu (built with -arch=compute_80 -rdc=true --ptx). Entry points,
// argument conventions and flag values are documented at the top of
// include/wgmma_emu/async_device.cuh.
//
// Every operation is executed synchronously by the issuing thread: copies and reductions are done
// with generic-proxy loads/stores/atomics, then the complete-tx (mbarrier completion) or bulk-group
// bookkeeping follows. This is one valid execution of the asynchronous operation (it completes
// before the program waits for it). Cluster size is 1, so every .shared::cluster address is a
// .shared::cta address of the executing CTA (the translator maps it before the call).
#include <stdio.h>

#include "wgmma_emu/async_device.cuh"

namespace wgemu {
namespace async {

WGEMU_DEV uint32_t linear_tid() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ __noinline__ void report(const char* kind, uint32_t a, uint32_t b, uint32_t flags) {
  printf("WGEMU-VIOLATION %s tid=%u block=(%u,%u,%u) a=0x%08x b=0x%08x\n", kind, linear_tid(),
         blockIdx.x, blockIdx.y, blockIdx.z, a, b);
  if (flags & WGEMU_AF_TRAP) __trap();
}

__device__ __noinline__ void fatal(const char* what, uint32_t a, uint32_t b) {
  printf("WGEMU-ERROR async %s tid=%u block=(%u,%u,%u) a=0x%08x b=0x%08x\n", what, linear_tid(),
         blockIdx.x, blockIdx.y, blockIdx.z, a, b);
  __trap();
}

// Undefined behaviour of the emulated program: check mode reports a violation and continues with a
// clamped value, the other modes stop the kernel.
WGEMU_DEV void undefined(const char* kind, uint32_t a, uint32_t b, uint32_t flags) {
  if (flags & WGEMU_AF_CHECK) {
    report(kind, a, b, flags);
  } else {
    fatal(kind, a, b);
  }
}

// ==== mbarrier ===================================================================================
constexpr uint32_t kMbErrUninit = 1u;
constexpr uint32_t kMbErrUnderflow = 2u;
constexpr uint32_t kMbErrTx = 4u;
constexpr uint32_t kMbErrDrop = 8u;
constexpr uint32_t kMbErrNoComplete = 16u;

__device__ __noinline__ void mb_report(uint32_t err, uint32_t bar, uint64_t old, uint32_t flags) {
  const uint32_t hi = (uint32_t)(old >> 32);
  if ((err & kMbErrUninit) && (flags & WGEMU_AF_CHECK))
    report("mbarrier-uninitialized", bar, hi, flags);
  if (err & kMbErrUnderflow) undefined("mbarrier-pending-underflow", bar, mb_pending(old), flags);
  if (err & kMbErrTx) undefined("mbarrier-tx-count-range", bar, (uint32_t)mb_tx(old), flags);
  if (err & kMbErrDrop) undefined("mbarrier-expected-count-zero", bar, mb_expected(old), flags);
  if (err & kMbErrNoComplete) undefined("mbarrier-noComplete-completed-phase", bar, hi, flags);
}

// One atomic update of the barrier word: arrive-on with `arrive` (0 = none), tx-count += dtx,
// optional expected-count drop, then phase completion (pending == 0 and tx-count == 0, ptx.txt
// 21043-21050). The completion check runs on every update, so a complete-tx that brings the
// tx-count to zero after all arrivals completes the phase [inferred]. Returns the previous word.
WGEMU_DEV uint64_t mb_update(uint32_t bar, uint32_t arrive, int64_t dtx, bool drop,
                             bool no_complete, uint32_t flags) {
  uint64_t old = mb_load_relaxed(bar);
  uint32_t err;
  for (;;) {
    err = mb_valid(old) ? 0u : kMbErrUninit;
    const uint32_t pend = mb_pending(old);
    uint32_t exp = mb_expected(old);
    int64_t tx = (int64_t)mb_tx(old) + dtx;
    if (tx > WGEMU_MB_TX_LIMIT || tx < -WGEMU_MB_TX_LIMIT) {
      err |= kMbErrTx;
      tx = tx > 0 ? WGEMU_MB_TX_LIMIT : -WGEMU_MB_TX_LIMIT;
    }
    uint32_t np;
    if (arrive > pend) {
      err |= kMbErrUnderflow;
      np = 0;
    } else {
      np = pend - arrive;
    }
    if (drop) {
      if (arrive >= exp) {
        err |= kMbErrDrop;
        exp = 0;
      } else {
        exp -= arrive;
      }
    }
    uint32_t ph = mb_phase(old);
    if (np == 0 && tx == 0) {
      if (no_complete) err |= kMbErrNoComplete;
      ph ^= 1u;
      np = exp;
    }
    const uint64_t keep = old & ((1ull << WGEMU_MB_VALID_BIT) | (1ull << WGEMU_MB_V1_BIT));
    const uint64_t nw = mb_pack(np, exp, (int32_t)tx, ph, keep);
    const uint64_t prev = mb_cas(bar, old, nw);
    if (prev == old) break;
    old = prev;
  }
  if (err) mb_report(err, bar, old, flags);
  return old;
}

WGEMU_DEV uint32_t mb_state(uint64_t old) { return (mb_phase(old) << 31) | mb_pending(old); }

WGEMU_DEV void mb_check_valid(uint32_t bar, uint64_t w, uint32_t flags) {
  if ((flags & WGEMU_AF_CHECK) && !mb_valid(w)) {
    report("mbarrier-uninitialized", bar, (uint32_t)(w >> 32), flags);
  }
}

// Returns the previous barrier word (E1 binds a delayed copy to its phase parity).
WGEMU_DEV uint64_t complete_tx(uint32_t bar, uint32_t bytes, uint32_t flags) {
  return mb_update(bar, 0u, -(int64_t)bytes, false, false, flags);
}

WGEMU_DEV uint32_t phase_done(uint32_t bar, uint32_t parity, uint32_t flags) {
  const uint64_t w = mb_load_acquire(bar);
  mb_check_valid(bar, w, flags);
  return mb_phase(w) != (parity & 1u) ? 1u : 0u;
}

// Default try_wait suspension limit when the program gives no timeHint [inferred: the hardware
// limit is system dependent; any finite limit is a valid implementation].
constexpr uint32_t kTryWaitDefaultNs = 20000u;

WGEMU_DEV uint32_t try_phase(uint32_t bar, uint32_t parity, uint32_t hint_ns, uint32_t flags) {
  const uint32_t limit = (flags & WGEMU_AF_MB_HINT) ? hint_ns : kTryWaitDefaultNs;
  uint64_t t0 = 0;
  bool started = false;
  for (;;) {
    if (phase_done(bar, parity, flags)) return 1u;
    const uint64_t now = globaltimer();
    if (!started) {
      t0 = now;
      started = true;
    } else if (now - t0 >= limit) {
      return 0u;
    }
    nanosleep_ns(64);
  }
}

// ==== tensor maps ================================================================================
struct Tmap {
  uint64_t gaddr;
  uint32_t misc;
  int rank, etype, esize, interleave, swizzle, atomicity, im2col;
  uint64_t dim[5];
  uint64_t stride[5];  // bytes; stride[0] is the element size
  uint32_t box[5], es[5];
  uint32_t cpp, ppc;
  int32_t lower[3], upper[3];
};

WGEMU_DEV uint32_t byte_at(const uint32_t* w, int off) {
  return (w[off >> 2] >> ((off & 3) * 8)) & 0xFFu;
}

WGEMU_DEV void tmap_read(uint64_t p, Tmap& t) {
  if (p & 15) fatal("tensormap-misaligned", (uint32_t)p, (uint32_t)(p >> 32));
  uint32_t w[32];
#pragma unroll
  for (int i = 0; i < 8; ++i) gen_ld16(p + 16 * i, w + 4 * i);
  t.misc = w[2];
  if (w[3] != WGEMU_TMAP_MAGIC) fatal("tensormap-not-encoded-by-wgemu-shim", w[3], w[2]);
  t.gaddr = (uint64_t)w[0] | ((uint64_t)w[1] << 32);
  t.rank = (int)wgemu_tm_get(t.misc, WGEMU_TM_RANK_M1_SHIFT, 3) + 1;
  t.etype = (int)wgemu_tm_get(t.misc, WGEMU_TM_ELEMTYPE_SHIFT, 4);
  t.interleave = (int)wgemu_tm_get(t.misc, WGEMU_TM_INTERLEAVE_SHIFT, 2);
  t.swizzle = (int)wgemu_tm_get(t.misc, WGEMU_TM_SWIZZLE_SHIFT, 3);
  t.atomicity = (int)wgemu_tm_get(t.misc, WGEMU_TM_ATOMICITY_SHIFT, 2);
  t.im2col = (int)wgemu_tm_get(t.misc, WGEMU_TM_IM2COL_SHIFT, 1);
  t.esize = wgemu_tm_elem_bytes(t.etype);
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    t.dim[i] = (uint64_t)w[4 + i] + 1u;
    t.box[i] = byte_at(w, 72 + i) + 1u;
    t.es[i] = byte_at(w, 80 + i) + 1u;
  }
  t.stride[0] = (uint64_t)t.esize;
#pragma unroll
  for (int i = 1; i < 5; ++i) {
    t.stride[i] = (uint64_t)w[10 + 2 * (i - 1)] | ((uint64_t)w[11 + 2 * (i - 1)] << 32);
  }
  t.cpp = byte_at(w, 85) + 1u;
  t.ppc = (byte_at(w, 86) | (byte_at(w, 87) << 8)) + 1u;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    t.lower[i] = (int32_t)(int16_t)(byte_at(w, 88 + 2 * i) | (byte_at(w, 89 + 2 * i) << 8));
    t.upper[i] = (int32_t)(int16_t)(byte_at(w, 94 + 2 * i) | (byte_at(w, 95 + 2 * i) << 8));
  }
  if (t.rank > 5) fatal("tensormap-rank", (uint32_t)t.rank, t.misc);
  if (t.esize == 0) fatal("tensormap-elemtype-unsupported", (uint32_t)t.etype, t.misc);
  if (t.swizzle > WGEMU_TM_SWIZZLE_128B || t.atomicity != WGEMU_TM_ATOM_16B) {
    fatal("tensormap-swizzle-unsupported", (uint32_t)t.swizzle, (uint32_t)t.atomicity);
  }
}

// Reduction element type implied by a tensor-map element type for cp.reduce.async.bulk.tensor.
WGEMU_DEV uint32_t tm_red_type(int etype) {
  switch (etype) {
    case WGEMU_TM_U32:
      return WGEMU_RT_U32;
    case WGEMU_TM_S32:
      return WGEMU_RT_S32;
    case WGEMU_TM_U64:
      return WGEMU_RT_U64;
    case WGEMU_TM_S64:
      return WGEMU_RT_S64;
    case WGEMU_TM_F16:
      return WGEMU_RT_F16;
    case WGEMU_TM_BF16:
      return WGEMU_RT_BF16;
    case WGEMU_TM_F32:
      return WGEMU_RT_F32;
    case WGEMU_TM_F32_FTZ:
      return WGEMU_RT_F32_FTZ;
    default:
      return 0xFFu;
  }
}

// Valid redOp / type pairs of cp.reduce.async.bulk.tensor (ptx.txt 16515-16533).
WGEMU_DEV bool tensor_red_ok(uint32_t op, uint32_t ty) {
  const bool i32 = ty == WGEMU_RT_U32 || ty == WGEMU_RT_S32;
  const bool i64 = ty == WGEMU_RT_U64 || ty == WGEMU_RT_S64;
  const bool h16 = ty == WGEMU_RT_F16 || ty == WGEMU_RT_BF16;
  switch (op) {
    case WGEMU_RED_ADD:
      return i32 || ty == WGEMU_RT_U64 || ty == WGEMU_RT_F32 || ty == WGEMU_RT_F32_FTZ || h16;
    case WGEMU_RED_MIN:
    case WGEMU_RED_MAX:
      return i32 || i64 || h16;
    case WGEMU_RED_INC:
    case WGEMU_RED_DEC:
      return ty == WGEMU_RT_U32;
    case WGEMU_RED_AND:
    case WGEMU_RED_OR:
    case WGEMU_RED_XOR:
      return i32 || i64;
    default:
      return false;
  }
}

// ==== reductions =================================================================================
WGEMU_DEV uint32_t red_bytes(uint32_t ty) {
  switch (ty) {
    case WGEMU_RT_B64:
    case WGEMU_RT_U64:
    case WGEMU_RT_S64:
    case WGEMU_RT_F64:
      return 8u;
    case WGEMU_RT_F16:
    case WGEMU_RT_BF16:
      return 2u;
    default:
      return 4u;
  }
}

WGEMU_DEV uint16_t op16(uint32_t op, bool bf, uint16_t a, uint16_t b) {
  uint16_t r = 0;
  const uint16_t one = 0x3F80u;  // bf16 1.0
  if (op == WGEMU_RED_ADD) {
    if (bf) {
      asm("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(r) : "h"(a), "h"(one), "h"(b));
    } else {
      asm("add.rn.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
    }
  } else if (op == WGEMU_RED_MIN) {
    if (bf) {
      asm("min.bf16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
    } else {
      asm("min.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
    }
  } else {
    if (bf) {
      asm("max.bf16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
    } else {
      asm("max.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
    }
  }
  return r;
}

WGEMU_DEV uint32_t add_f32(uint32_t a, uint32_t b, bool ftz) {
  float r;
  if (ftz) {
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a)), "f"(__uint_as_float(b)));
  } else {
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a)), "f"(__uint_as_float(b)));
  }
  return __float_as_uint(r);
}

// Element-wise reduction into global memory (generic address a). Scope .sys covers every scope a
// program can request (.cta/.cluster/.gpu/.sys); relaxed semantics as the instruction requires.
__device__ __noinline__ void red_global(uint64_t a, uint32_t op, uint32_t ty, uint64_t v) {
  const uint64_t g = to_global(a);
  switch (ty) {
    case WGEMU_RT_B32:
    case WGEMU_RT_U32:
    case WGEMU_RT_S32: {
      const uint32_t x = (uint32_t)v;
      const bool s = ty == WGEMU_RT_S32;
      switch (op) {
        case WGEMU_RED_ADD:
          asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        case WGEMU_RED_MIN:
          if (s) {
            asm volatile("red.relaxed.sys.global.min.s32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          } else {
            asm volatile("red.relaxed.sys.global.min.u32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          }
          return;
        case WGEMU_RED_MAX:
          if (s) {
            asm volatile("red.relaxed.sys.global.max.s32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          } else {
            asm volatile("red.relaxed.sys.global.max.u32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          }
          return;
        case WGEMU_RED_INC:
          asm volatile("red.relaxed.sys.global.inc.u32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        case WGEMU_RED_DEC:
          asm volatile("red.relaxed.sys.global.dec.u32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        case WGEMU_RED_AND:
          asm volatile("red.relaxed.sys.global.and.b32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        case WGEMU_RED_OR:
          asm volatile("red.relaxed.sys.global.or.b32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        case WGEMU_RED_XOR:
          asm volatile("red.relaxed.sys.global.xor.b32 [%0], %1;" ::"l"(g), "r"(x) : "memory");
          return;
        default:
          break;
      }
      break;
    }
    case WGEMU_RT_B64:
    case WGEMU_RT_U64:
    case WGEMU_RT_S64: {
      const bool s = ty == WGEMU_RT_S64;
      switch (op) {
        case WGEMU_RED_ADD:
          asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          return;
        case WGEMU_RED_MIN:
          if (s) {
            asm volatile("red.relaxed.sys.global.min.s64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          } else {
            asm volatile("red.relaxed.sys.global.min.u64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          }
          return;
        case WGEMU_RED_MAX:
          if (s) {
            asm volatile("red.relaxed.sys.global.max.s64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          } else {
            asm volatile("red.relaxed.sys.global.max.u64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          }
          return;
        case WGEMU_RED_AND:
          asm volatile("red.relaxed.sys.global.and.b64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          return;
        case WGEMU_RED_OR:
          asm volatile("red.relaxed.sys.global.or.b64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          return;
        case WGEMU_RED_XOR:
          asm volatile("red.relaxed.sys.global.xor.b64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
          return;
        default:
          break;
      }
      break;
    }
    case WGEMU_RT_F32:
    case WGEMU_RT_F32_FTZ: {
      if (op != WGEMU_RED_ADD) break;
      // atom.add.f32 on global flushes subnormals on sm_80 and .noftz needs sm_90: CAS loop.
      const bool ftz = ty == WGEMU_RT_F32_FTZ;
      uint32_t old;
      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(old) : "l"(g) : "memory");
      for (;;) {
        const uint32_t nv = add_f32(old, (uint32_t)v, ftz);
        uint32_t prev;
        asm volatile("atom.relaxed.sys.global.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(prev)
                     : "l"(g), "r"(old), "r"(nv)
                     : "memory");
        if (prev == old) return;
        old = prev;
      }
    }
    case WGEMU_RT_F64:
      if (op != WGEMU_RED_ADD) break;
      asm volatile("red.relaxed.sys.global.add.f64 [%0], %1;" ::"l"(g),
                   "d"(__longlong_as_double((long long)v))
                   : "memory");
      return;
    case WGEMU_RT_F16:
      if (op == WGEMU_RED_ADD) {
        asm volatile("red.relaxed.sys.global.add.noftz.f16 [%0], %1;" ::"l"(g), "h"((uint16_t)v)
                     : "memory");
        return;
      }
      [[fallthrough]];
    case WGEMU_RT_BF16: {
      if (op != WGEMU_RED_ADD && op != WGEMU_RED_MIN && op != WGEMU_RED_MAX) break;
      const bool bf = ty == WGEMU_RT_BF16;
      uint16_t old;
      asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(old) : "l"(g) : "memory");
      for (;;) {
        const uint16_t nv = op16(op, bf, old, (uint16_t)v);
        uint16_t prev;
        asm volatile("atom.relaxed.sys.global.cas.b16 %0, [%1], %2, %3;"
                     : "=h"(prev)
                     : "l"(g), "h"(old), "h"(nv)
                     : "memory");
        if (prev == old) return;
        old = prev;
      }
    }
    default:
      break;
  }
  fatal("reduce-op-type-unsupported", op, ty);
}

// Element-wise reduction into the executing CTA's shared memory (cluster size 1).
__device__ __noinline__ void red_shared(uint32_t a, uint32_t op, uint32_t ty, uint64_t v) {
  const uint32_t x = (uint32_t)v;
  const bool s = ty == WGEMU_RT_S32;
  const sh_addr_t sa = sh_addr(a, ty == WGEMU_RT_U64 ? 8u : 4u);
  if (ty == WGEMU_RT_B32 || ty == WGEMU_RT_U32 || ty == WGEMU_RT_S32) {
    switch (op) {
      case WGEMU_RED_ADD:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".add.u32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      case WGEMU_RED_MIN:
        if (s) {
          asm volatile("red.relaxed.cta." WGEMU_SH_SP ".min.s32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                       : "memory");
        } else {
          asm volatile("red.relaxed.cta." WGEMU_SH_SP ".min.u32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                       : "memory");
        }
        return;
      case WGEMU_RED_MAX:
        if (s) {
          asm volatile("red.relaxed.cta." WGEMU_SH_SP ".max.s32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                       : "memory");
        } else {
          asm volatile("red.relaxed.cta." WGEMU_SH_SP ".max.u32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                       : "memory");
        }
        return;
      case WGEMU_RED_INC:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".inc.u32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      case WGEMU_RED_DEC:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".dec.u32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      case WGEMU_RED_AND:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".and.b32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      case WGEMU_RED_OR:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".or.b32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      case WGEMU_RED_XOR:
        asm volatile("red.relaxed.cta." WGEMU_SH_SP ".xor.b32 [%0], %1;" ::WGEMU_SH_C(sa), "r"(x)
                     : "memory");
        return;
      default:
        break;
    }
  } else if (ty == WGEMU_RT_U64 && op == WGEMU_RED_ADD) {
    asm volatile("red.relaxed.cta." WGEMU_SH_SP ".add.u64 [%0], %1;" ::WGEMU_SH_C(sa), "l"(v)
                 : "memory");
    return;
  }
  fatal("reduce-op-type-unsupported-shared", op, ty);
}

// ==== check-mode bulk-group table ================================================================
// Layout at the generic address `st` (function-scope .local buffer, WGEMU_BULK_STATE_BYTES):
//   u32 committed group count, u32 entry count (bit 31: table-full warning printed), then
//   WGEMU_BULK_MAX entries {u32 smem addr, u32 bytes, u32 hash, u32 group}.
WGEMU_DEV uint32_t smem_hash(uint32_t a, uint32_t n) {
  uint32_t h = 2166136261u;
  if (((a | n) & 3u) == 0u) {
    for (uint32_t i = 0; i < n; i += 4) {
      h = (h ^ (uint32_t)shared_ld(a + i, 4)) * 16777619u;
    }
  } else {
    for (uint32_t i = 0; i < n; ++i) {
      h = (h ^ (uint32_t)shared_ld(a + i, 1)) * 16777619u;
    }
  }
  return h;
}

__device__ __noinline__ void bulk_record(uint64_t st, uint32_t addr, uint32_t bytes) {
  if (st == 0 || bytes == 0) return;
  const uint32_t cur = (uint32_t)gen_ld(st, 4);
  const uint32_t cnt_raw = (uint32_t)gen_ld(st + 4, 4);
  const uint32_t cnt = cnt_raw & 0xFFFFu;
  if (cnt >= WGEMU_BULK_MAX) {
    if (!(cnt_raw & 0x80000000u)) {
      printf("WGEMU-WARNING async bulk-check-table-full tid=%u: further bulk sources unchecked\n",
             linear_tid());
      gen_st(st + 4, 4, cnt_raw | 0x80000000u);
    }
    return;
  }
  const uint64_t e = st + 8 + 16ull * cnt;
  gen_st(e, 4, addr);
  gen_st(e + 4, 4, bytes);
  gen_st(e + 8, 4, smem_hash(addr, bytes));
  gen_st(e + 12, 4, cur);
  gen_st(st + 4, 4, (cnt_raw & 0x80000000u) | (cnt + 1u));
}

// Source range read by a tensor store/reduce from shared memory: with a swizzle mode the bytes
// are permuted inside 128-byte lines, so the whole lines are covered [inferred].
WGEMU_DEV void tensor_src_range(uint32_t src, uint32_t bytes, int swizzle, uint32_t& lo,
                                uint32_t& n) {
  if (swizzle == WGEMU_TM_SWIZZLE_NONE) {
    lo = src;
    n = bytes;
  } else {
    lo = src & ~127u;
    n = ((src + bytes + 127u) & ~127u) - lo;
  }
}

// ==== element copy ===============================================================================
constexpr int kLoad = 0;
constexpr int kStore = 1;
constexpr int kReduce = 2;

// E1 sink of a delayed copy ("E1 delayed delivery" in async_device.cuh): the data goes to the
// shadow buffer, WGEMU_DL_POISON to the destination words, and the owner map takes the record
// index + 1 of every destination word. `overlap` is 1 + the first byte address of a word that
// belonged to another pending copy; [lo, hi) is the word range written.
struct Sink {
  uint64_t shadow, owner;
  uint32_t idx1, overlap, lo, hi;
};

WGEMU_DEV void sink_range(Sink& s, uint32_t a, uint32_t n) {
  if ((uint64_t)a + n > WGEMU_DL_SMEM_BYTES) fatal("async-delay-smem-address-range", a, n);
  const uint32_t lo = a & ~3u;
  const uint32_t hi = (a + n + 3u) & ~3u;
  if (lo < s.lo) s.lo = lo;
  if (hi > s.hi) s.hi = hi;
}

// Takes word w (byte address 4w) for the copy. A word first taken by this copy keeps its current
// bytes in the shadow (the copy may write part of it), unless another pending copy owns it: then
// the shadow already holds that copy's data and the overlap is recorded.
WGEMU_DEV void sink_take(Sink& s, uint32_t w) {
  const uint64_t o = s.owner + 2ull * w;
  const uint32_t prev = (uint32_t)gen_ld(o, 2);
  if (prev == s.idx1) return;
  if (prev != 0u) {
    if (s.overlap == 0u) s.overlap = (w << 2) + 1u;
  } else {
    gen_st(s.shadow + 4ull * w, 4, shared_ld(w << 2, 4));
  }
  gen_st(o, 2, s.idx1);
  shared_st(w << 2, 4, WGEMU_DL_POISON);
}

// 16-byte aligned chunk: four whole words.
WGEMU_DEV void sink_put16(Sink& s, uint32_t a, const uint32_t* w) {
  sink_range(s, a, 16u);
  const uint64_t o = s.owner + 2ull * (a >> 2);
  const uint64_t prev = gen_ld(o, 8);
  const uint64_t mine = 0x0001000100010001ull * s.idx1;
  if (prev != 0ull && prev != mine && s.overlap == 0u) {
    for (uint32_t k = 0; k < 4u; ++k) {
      const uint32_t p = (uint32_t)(prev >> (16u * k)) & 0xFFFFu;
      if (p != 0u && p != s.idx1) {
        s.overlap = a + 4u * k + 1u;
        break;
      }
    }
  }
  gen_st(o, 8, mine);
  gen_st16(s.shadow + a, w);
  const uint32_t poison[4] = {WGEMU_DL_POISON, WGEMU_DL_POISON, WGEMU_DL_POISON, WGEMU_DL_POISON};
  shared_st16(a, poison);
}

WGEMU_DEV void sink_put(Sink& s, uint32_t a, int n, uint64_t v) {
  sink_range(s, a, (uint32_t)n);
  for (uint32_t w = a >> 2; w <= (a + (uint32_t)n - 1u) >> 2; ++w) sink_take(s, w);
  gen_st(s.shadow + a, n, v);
}

// Destination store of a copy into shared memory: direct, or into the E1 sink.
WGEMU_DEV void dst_st16(Sink* s, uint32_t a, const uint32_t* w) {
  if (s != nullptr) {
    sink_put16(*s, a, w);
  } else {
    shared_st16(a, w);
  }
}

WGEMU_DEV void dst_st(Sink* s, uint32_t a, int n, uint64_t v) {
  if (s != nullptr) {
    sink_put(*s, a, n, v);
  } else {
    shared_st(a, n, v);
  }
}

struct Xfer {
  int kind;
  int esize;
  int etype;
  int swizzle;
  bool convert;
  uint64_t fill;
  uint32_t red_op, red_type;
  Sink* sink;  // loads only: E1 delayed copy (nullptr: shared memory is written directly)
};

// One contiguous shared-memory row of n elements. Global element k is at grow + x_k * esize with
// x_k = x0 + k * step; it is in bounds when row_in and 0 <= x_k < dim0. Loads fill out-of-bounds
// elements, stores and reductions skip them.
WGEMU_DEV void xfer_row(const Xfer& x, uint64_t grow, bool row_in, int64_t x0, uint32_t step,
                        uint32_t n, uint64_t dim0, uint32_t srow) {
  const int eb = x.esize;
  const uint32_t row_bytes = n * (uint32_t)eb;
  const bool full = row_in && step == 1u && x0 >= 0 && (uint64_t)x0 + n <= dim0;
  if (full && !x.convert && x.kind != kReduce && ((row_bytes | srow) & 15u) == 0u &&
      ((grow + (uint64_t)x0 * eb) & 15u) == 0u) {
    const uint64_t g = grow + (uint64_t)x0 * eb;
    uint32_t w[4];
    for (uint32_t o = 0; o < row_bytes; o += 16) {
      const uint32_t sa = wgemu_tm_swizzle_addr(srow + o, x.swizzle);
      if (x.kind == kLoad) {
        gen_ld16(g + o, w);
        dst_st16(x.sink, sa, w);
      } else {
        shared_ld16(sa, w);
        gen_st16(g + o, w);
      }
    }
    return;
  }
  for (uint32_t k = 0; k < n; ++k) {
    const int64_t xk = x0 + (int64_t)k * step;
    const bool in = row_in && xk >= 0 && (uint64_t)xk < dim0;
    const uint32_t sa = wgemu_tm_swizzle_addr(srow + k * (uint32_t)eb, x.swizzle);
    const uint64_t ga = grow + (uint64_t)xk * (uint64_t)eb;
    if (x.kind == kLoad) {
      uint64_t v = x.fill;
      if (in) {
        v = gen_ld(ga, eb);
        if (x.convert) v = tm_convert(v, x.etype);
      }
      dst_st(x.sink, sa, eb, v);
    } else if (in) {
      uint64_t v = shared_ld(sa, eb);
      if (x.convert) v = tm_convert(v, x.etype);
      if (x.kind == kStore) {
        gen_st(ga, eb, v);
      } else {
        red_global(ga, x.red_op, x.red_type, v);
      }
    }
  }
}

WGEMU_DEV Xfer make_xfer(int kind, const Tmap& t, uint32_t red_op) {
  Xfer x;
  x.kind = kind;
  x.esize = t.esize;
  x.etype = t.etype;
  x.swizzle = t.swizzle;
  x.convert = tm_needs_convert(t.etype);
  x.fill = (wgemu_tm_get(t.misc, WGEMU_TM_FILL_SHIFT, 1) && wgemu_tm_elem_is_float(t.etype))
               ? tm_nan_fill(t.etype)
               : 0ull;
  x.red_op = red_op;
  x.red_type = 0;
  x.sink = nullptr;
  if (kind == kReduce) {
    x.red_type = tm_red_type(t.etype);
    if (!tensor_red_ok(red_op, x.red_type)) {
      fatal("tensor-reduce-op-elemtype-unsupported", red_op, (uint32_t)t.etype);
    }
    x.convert = false;  // reductions use the element type itself (f32.ftz handled by red_global)
  }
  return x;
}

// Tile mode (ptx.txt 3210-3248, cuda.h 23305-23310): ceil(box[i] / elementStrides[i]) elements
// per dimension, dimension 0 ignores its element stride unless the layout is interleaved; the
// shared-memory box is dense with dimension 0 fastest. Returns the box size in bytes.
WGEMU_DEV uint32_t tile_xfer(const Tmap& t, const Xfer& x, uint32_t smem, const int32_t* c) {
  uint32_t n[5], step[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    if (i >= t.rank) {
      n[i] = 1u;
      step[i] = 1u;
    } else {
      step[i] = (i == 0 && t.interleave == WGEMU_TM_INTERLEAVE_NONE) ? 1u : t.es[i];
      n[i] = (t.box[i] + step[i] - 1u) / step[i];
    }
  }
  const uint32_t row_bytes = n[0] * (uint32_t)t.esize;
  const uint32_t rows = n[1] * n[2] * n[3] * n[4];
  for (uint32_t r = 0; r < rows; ++r) {
    uint32_t q = r;
    bool row_in = true;
    uint64_t grow = t.gaddr;
#pragma unroll
    for (int d = 1; d < 5; ++d) {
      const uint32_t k = q % n[d];
      q /= n[d];
      if (d < t.rank) {
        const int64_t xd = (int64_t)c[d] + (int64_t)k * step[d];
        if (xd < 0 || (uint64_t)xd >= t.dim[d]) {
          row_in = false;
        } else {
          grow += (uint64_t)xd * t.stride[d];
        }
      }
    }
    xfer_row(x, grow, row_in, c[0], step[0], n[0], t.dim[0], smem + r * row_bytes);
  }
  return rows * row_bytes;
}

// im2col mode (ptx.txt 3267-3406; cute/atom/copy_traits_sm90_im2col.hpp). Rank r tensor
// {C, W[, H[, D]], N}. Pixel p reads channels [c0, c0 + channelsPerPixel) at spatial position
// (w + offW, h + offH, d + offD) of image n. The traversal starts at the coordinates, steps W by
// its element stride and wraps to the bounding-box lower corner after dim - 1 + upper corner,
// carrying into H, D and then N. `off` holds the im2col offsets (zero for im2col_no_offs).
WGEMU_DEV uint32_t im2col_xfer(const Tmap& t, const Xfer& x, uint32_t smem, const int32_t* c,
                               const uint32_t* off) {
  const int ns = t.rank - 2;
  if (ns < 1) fatal("im2col-rank", (uint32_t)t.rank, 0u);
  int64_t pos[3] = {0, 0, 0};
  for (int s = 0; s < ns; ++s) pos[s] = c[1 + s];
  int64_t img = c[t.rank - 1];
  const uint32_t row_bytes = t.cpp * (uint32_t)t.esize;
  for (uint32_t p = 0; p < t.ppc; ++p) {
    bool in = img >= 0 && (uint64_t)img < t.dim[t.rank - 1];
    uint64_t grow = t.gaddr + (in ? (uint64_t)img * t.stride[t.rank - 1] : 0ull);
    for (int s = 0; s < ns; ++s) {
      const int64_t q = pos[s] + (int64_t)off[s];
      if (q < 0 || (uint64_t)q >= t.dim[1 + s]) {
        in = false;
      } else {
        grow += (uint64_t)q * t.stride[1 + s];
      }
    }
    xfer_row(x, grow, in, c[0], 1u, t.cpp, t.dim[0], smem + p * row_bytes);
    for (int s = 0; s < ns; ++s) {
      pos[s] += t.es[1 + s];
      if (pos[s] <= (int64_t)t.dim[1 + s] - 1 + t.upper[s]) break;
      pos[s] = t.lower[s];
      if (s == ns - 1) ++img;
    }
  }
  return t.ppc * row_bytes;
}

WGEMU_DEV void tensor_check_mode(const Tmap& t, uint32_t flags) {
  const uint32_t dim = flags & WGEMU_AF_DIM_MASK;
  const uint32_t im2col = (flags & WGEMU_AF_IM2COL) ? 1u : 0u;
  if (dim != (uint32_t)t.rank)
    fatal("tensor-dim-differs-from-tensormap-rank", dim, (uint32_t)t.rank);
  if (im2col != (uint32_t)t.im2col)
    fatal("tensor-load-mode-differs-from-tensormap", im2col, t.misc);
}

WGEMU_DEV void coords(int32_t* c, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t c4) {
  c[0] = (int32_t)c0;
  c[1] = (int32_t)c1;
  c[2] = (int32_t)c2;
  c[3] = (int32_t)c3;
  c[4] = (int32_t)c4;
}

// Multicast with cluster size 1: only CTA rank 0 (the executing CTA) exists. Returns whether the
// executing CTA is a destination; a mask naming other ranks is an error [inferred].
WGEMU_DEV bool multicast_self(uint32_t cta_mask, uint32_t flags) {
  if (!(flags & WGEMU_AF_MULTICAST)) return true;
  if (cta_mask & ~1u) fatal("multicast-to-missing-cta-rank", cta_mask, 1u);
  return (cta_mask & 1u) != 0u;
}

// ==== bulk copies ================================================================================
WGEMU_DEV void bulk_check_align(uint64_t a, uint64_t b, uint32_t size, uint32_t flags) {
  if (((a | b | size) & 15u) != 0u) {
    undefined("bulk-copy-misaligned", (uint32_t)(a | b), size, flags);
  }
}

// Tensor load into shared memory (tile or im2col) without the barrier update. Returns false when
// the executing CTA is not a destination of a multicast copy, else the box size in `bytes`.
WGEMU_DEV bool tma_load_run(uint32_t dst, uint64_t tmap, uint32_t c0, uint32_t c1, uint32_t c2,
                            uint32_t c3, uint32_t c4, uint32_t o0, uint32_t o1, uint32_t o2,
                            uint32_t cta_mask, uint32_t flags, Sink* sink, uint32_t& bytes) {
  Tmap t;
  tmap_read(tmap, t);
  tensor_check_mode(t, flags);
  if (!multicast_self(cta_mask, flags)) return false;
  if (dst & 15u) undefined("tensor-copy-smem-misaligned", dst, 0u, flags);
  int32_t c[5];
  coords(c, c0, c1, c2, c3, c4);
  Xfer x = make_xfer(kLoad, t, 0u);
  x.sink = sink;
  if (t.im2col) {
    const uint32_t off[3] = {o0 & 0xFFFFu, o1 & 0xFFFFu, o2 & 0xFFFFu};
    bytes = im2col_xfer(t, x, dst, c, off);
  } else {
    bytes = tile_xfer(t, x, dst, c);
  }
  return true;
}

// Bulk global -> shared copy without the barrier update (the multicast check is the caller's).
WGEMU_DEV void bulk_g2s_run(uint32_t dst, uint64_t src, uint32_t size, uint32_t ign_left,
                            uint32_t ign_right, uint32_t flags, Sink* sink) {
  bulk_check_align(dst, src, size, flags);
  if (ign_left > 15u || ign_right > 15u) {
    undefined("bulk-copy-ignore-bytes-range", ign_left, ign_right, flags);
  }
  const uint32_t lo = ign_left & 15u;
  const uint32_t hi = size - (ign_right & 15u);
  uint32_t w[4];
  for (uint32_t o = 0; o < size; o += 16) {
    if (o >= lo && o + 16u <= hi) {
      gen_ld16(src + o, w);
      dst_st16(sink, dst + o, w);
    } else {
      // Ignored bytes are indeterminate; they are left untouched.
      for (uint32_t b = o; b < o + 16u && b < size; ++b) {
        if (b >= lo && b < hi) dst_st(sink, dst + b, 1, gen_ld(src + b, 1));
      }
    }
  }
}

#ifndef WGEMU_SMEM_GLOBAL  // E1 delayed delivery is not part of the E4 build
// ==== E1 delayed delivery ========================================================================
// Header fields (byte offsets from the shared address `hdr`) and slot regions, see
// "E1 delayed delivery" in async_device.cuh. Every update of records, owner map, shadow, header
// counters and of destinations of pending copies happens under the header lock.
constexpr uint32_t kHdrSlot = 0u;
constexpr uint32_t kHdrLock = 8u;
constexpr uint32_t kHdrAlive = 12u;
constexpr uint32_t kHdrLive = 16u;
constexpr uint32_t kHdrHiwater = 20u;
constexpr uint32_t kHdrNrec = 24u;
constexpr uint32_t kHdrMagic = 28u;
constexpr uint32_t kRecLo = 0u;
constexpr uint32_t kRecSpan = 4u;
constexpr uint32_t kRecBar = 8u;
constexpr uint32_t kRecInfo = 12u;
constexpr uint32_t kRecGroup = 16u;
constexpr uint32_t kDlNone = 0xFFFFFFFFu;
// A CTA waits this long for a free pool slot (running CTAs release theirs when they exit) before
// it reports async-delay-no-slot and runs without delay [inferred: deadlock guard only].
constexpr uint64_t kDlSlotWaitNs = 60000000000ull;

WGEMU_DEV uint32_t sh_atom_add(uint32_t a, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(a), "r"(v) : "memory");
  return old;
}

WGEMU_DEV void dl_lock(uint32_t hdr) {
  for (;;) {
    uint32_t old;
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(old)
                 : "r"(hdr + kHdrLock), "r"(0u), "r"(1u)
                 : "memory");
    if (old == 0u) return;
    nanosleep_ns(32);
  }
}

WGEMU_DEV void dl_unlock(uint32_t hdr) {
  asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(hdr + kHdrLock), "r"(0u)
               : "memory");
}

WGEMU_DEV uint32_t hdr_ld(uint32_t hdr, uint32_t f) { return (uint32_t)shared_ld(hdr + f, 4); }
WGEMU_DEV void hdr_st(uint32_t hdr, uint32_t f, uint32_t v) { shared_st(hdr + f, 4, v); }

// Pool slot of the executing CTA (0: copies are not delayed).
WGEMU_DEV uint64_t dl_slot(uint32_t hdr) {
  if (hdr_ld(hdr, kHdrMagic) != WGEMU_DL_MAGIC) {
    fatal("async-delay-header-uninitialized", hdr, hdr_ld(hdr, kHdrMagic));
  }
  return shared_ld(hdr + kHdrSlot, 8);
}

WGEMU_DEV uint64_t dl_rec(uint64_t slot, uint32_t i) {
  return slot + WGEMU_DL_SLOT_RECORDS + (uint64_t)WGEMU_DL_REC_BYTES * i;
}

WGEMU_DEV uint64_t dl_shadow(uint64_t slot, uint32_t nrec) {
  return slot + WGEMU_DL_SLOT_RECORDS + (uint64_t)WGEMU_DL_REC_BYTES * nrec;
}

WGEMU_DEV uint64_t dl_tword(uint64_t slot, uint32_t region, uint32_t tid) {
  return slot + region + 4ull * tid;
}

// Free record index (under the lock); kDlNone after reporting an overflow.
WGEMU_DEV uint32_t dl_alloc(uint32_t hdr, uint64_t slot, uint32_t flags) {
  const uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  for (uint32_t i = 0; i < hw; ++i) {
    if (!((uint32_t)gen_ld(dl_rec(slot, i) + kRecInfo, 4) & WGEMU_DL_LIVE)) return i;
  }
  const uint32_t nrec = hdr_ld(hdr, kHdrNrec);
  if (hw < nrec) {
    hdr_st(hdr, kHdrHiwater, hw + 1u);
    return hw;
  }
  report("async-delay-record-overflow", nrec, hdr_ld(hdr, kHdrLive), flags);
  return kDlNone;
}

WGEMU_DEV void dl_sink(uint32_t hdr, uint64_t slot, uint32_t idx, Sink& s) {
  s.shadow = dl_shadow(slot, hdr_ld(hdr, kHdrNrec));
  s.owner = slot + WGEMU_DL_SLOT_OWNER;
  s.idx1 = idx + 1u;
  s.overlap = 0u;
  s.lo = 0xFFFFFFFFu;
  s.hi = 0u;
}

// Makes record idx live after its data went into the sink (under the lock).
WGEMU_DEV void dl_commit(uint32_t hdr, uint64_t slot, uint32_t idx, const Sink& s, uint32_t bar,
                         uint32_t info, uint32_t group, uint32_t flags) {
  if (s.overlap != 0u) report("async-delay-overlap", s.overlap - 1u, bar, flags);
  const uint64_t r = dl_rec(slot, idx);
  gen_st(r + kRecLo, 4, s.hi > s.lo ? s.lo : 0u);
  gen_st(r + kRecSpan, 4, s.hi > s.lo ? s.hi - s.lo : 0u);
  gen_st(r + kRecBar, 4, bar);
  gen_st(r + kRecGroup, 4, group);
  gen_st(r + kRecInfo, 4, info | WGEMU_DL_LIVE);
  hdr_st(hdr, kHdrLive, hdr_ld(hdr, kHdrLive) + 1u);
}

// Writes the data of record i into shared memory and frees it (under the lock). Destination words
// that no longer hold WGEMU_DL_POISON were written by the program while the copy was pending.
WGEMU_DEV void dl_deliver(uint32_t hdr, uint64_t slot, uint32_t i, uint32_t flags) {
  const uint64_t r = dl_rec(slot, i);
  const uint32_t lo = (uint32_t)gen_ld(r + kRecLo, 4);
  const uint32_t span = (uint32_t)gen_ld(r + kRecSpan, 4);
  const uint32_t info = (uint32_t)gen_ld(r + kRecInfo, 4);
  const uint64_t shadow = dl_shadow(slot, hdr_ld(hdr, kHdrNrec));
  const uint64_t owner = slot + WGEMU_DL_SLOT_OWNER;
  bool written = false;
  for (uint32_t a = lo; a < lo + span; a += 4u) {
    const uint64_t o = owner + 2ull * (a >> 2);
    if ((uint32_t)gen_ld(o, 2) != i + 1u) continue;
    const uint32_t cur = (uint32_t)shared_ld(a, 4);
    if (cur != WGEMU_DL_POISON && !written) {
      report("async-delay-destination-written", a, cur, flags);
      written = true;
    }
    shared_st(a, 4, gen_ld(shadow + a, 4));
    gen_st(o, 2, 0ull);
  }
  gen_st(r + kRecInfo, 4, 0ull);
  hdr_st(hdr, kHdrLive, hdr_ld(hdr, kHdrLive) - 1u);
  if (info & WGEMU_DL_CP) {
    const uint64_t tc = dl_tword(slot, WGEMU_DL_SLOT_TCOUNT, info >> WGEMU_DL_OWNER_SHIFT);
    gen_st(tc, 4, gen_ld(tc, 4) - 1u);
  }
}

WGEMU_DEV void dl_shrink(uint32_t hdr, uint64_t slot) {
  uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  while (hw > 0u && !((uint32_t)gen_ld(dl_rec(slot, hw - 1u) + kRecInfo, 4) & WGEMU_DL_LIVE)) --hw;
  hdr_st(hdr, kHdrHiwater, hw);
}

// Delivers every pending copy bound to (bar, parity): a thread observed that phase complete.
WGEMU_DEV void dl_deliver_phase(uint32_t hdr, uint32_t bar, uint32_t parity, uint32_t flags) {
  const uint64_t slot = dl_slot(hdr);
  if (slot == 0ull || hdr_ld(hdr, kHdrLive) == 0u) return;
  dl_lock(hdr);
  const uint32_t want = WGEMU_DL_LIVE | WGEMU_DL_BOUND | ((parity & 1u) ? WGEMU_DL_PARITY : 0u);
  const uint32_t mask = WGEMU_DL_LIVE | WGEMU_DL_BOUND | WGEMU_DL_PARITY;
  const uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  for (uint32_t i = 0; i < hw; ++i) {
    const uint64_t r = dl_rec(slot, i);
    if (((uint32_t)gen_ld(r + kRecInfo, 4) & mask) == want &&
        (uint32_t)gen_ld(r + kRecBar, 4) == bar) {
      dl_deliver(hdr, slot, i, flags);
    }
  }
  dl_shrink(hdr, slot);
  dl_unlock(hdr);
}

#endif  // !WGEMU_SMEM_GLOBAL

WGEMU_DEV uint32_t warp_lane() {
  uint32_t l;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
  return l;
}

// ==== E2 counters ================================================================================
// Box size in bytes of one tensor copy (tile: product of the per-dimension element counts, im2col:
// pixels x channels), the value tile_xfer / im2col_xfer return.
WGEMU_DEV uint32_t tensor_box_bytes(const Tmap& t) {
  if (t.im2col) return t.ppc * t.cpp * (uint32_t)t.esize;
  uint32_t n = 1u;
  for (int i = 0; i < t.rank; ++i) {
    const uint32_t step = (i == 0 && t.interleave == WGEMU_TM_INTERLEAVE_NONE) ? 1u : t.es[i];
    n *= (t.box[i] + step - 1u) / step;
  }
  return n * (uint32_t)t.esize;
}

}  // namespace async
}  // namespace wgemu

extern "C" {

// ---- mbarrier -----------------------------------------------------------------------------------
__device__ __noinline__ void wgemu_mbar_init(uint32_t bar, uint32_t count, uint32_t flags) {
  using namespace wgemu::async;
  const bool v1 = (flags & WGEMU_AF_MB_V1) != 0u;
  const uint32_t maxc = v1 ? WGEMU_MB_COUNT_MAX_V1 : WGEMU_MB_COUNT_MAX_V0;
  if (bar & 7u) fatal("mbarrier-misaligned", bar, 0u);
  if (count == 0u || count > maxc) {
    undefined("mbarrier-init-count-range", bar, count, flags);
    count = count == 0u ? 1u : maxc;
  }
  const uint64_t keep = (1ull << WGEMU_MB_VALID_BIT) | (v1 ? (1ull << WGEMU_MB_V1_BIT) : 0ull);
  shared_st(bar, 8, mb_pack(count, count, 0, 0u, keep));
}

__device__ __noinline__ void wgemu_mbar_inval(uint32_t bar, uint32_t flags) {
  using namespace wgemu::async;
  mb_check_valid(bar, mb_load_relaxed(bar), flags);
  shared_st(bar, 8, 0ull);
}

__device__ __noinline__ uint32_t wgemu_mbar_arrive(uint32_t bar, uint32_t count, uint32_t tx,
                                                   uint32_t flags) {
  using namespace wgemu::async;
  // count range: Table 43 (ptx.txt 20915-20925) via ptx.txt 21379; 0 is outside it [inferred].
  // A count above the pending count of a v1 barrier is caught as an underflow by mb_update.
  if (count == 0u || count > WGEMU_MB_COUNT_MAX_V0) {
    undefined("mbarrier-arrive-count-range", bar, count, flags);
  }
  const int64_t dtx = (flags & WGEMU_AF_MB_EXPECT_TX) ? (int64_t)tx : 0;
  const uint64_t old = mb_update(bar, count, dtx, (flags & WGEMU_AF_MB_DROP) != 0u,
                                 (flags & WGEMU_AF_MB_NOCOMPLETE) != 0u, flags);
  return mb_state(old);
}

__device__ __noinline__ void wgemu_mbar_expect_tx(uint32_t bar, uint32_t tx, uint32_t flags) {
  wgemu::async::mb_update(bar, 0u, (int64_t)tx, false, false, flags);
}

__device__ __noinline__ void wgemu_mbar_complete_tx(uint32_t bar, uint32_t tx, uint32_t flags) {
  wgemu::async::complete_tx(bar, tx, flags);
}

__device__ __noinline__ uint32_t wgemu_mbar_test_wait(uint32_t bar, uint64_t state,
                                                      uint32_t flags) {
  return wgemu::async::phase_done(bar, (uint32_t)(state >> 31), flags);
}

__device__ __noinline__ uint32_t wgemu_mbar_test_wait_parity(uint32_t bar, uint32_t parity,
                                                             uint32_t flags) {
  return wgemu::async::phase_done(bar, parity, flags);
}

__device__ __noinline__ uint32_t wgemu_mbar_try_wait(uint32_t bar, uint64_t state, uint32_t hint_ns,
                                                     uint32_t flags) {
  return wgemu::async::try_phase(bar, (uint32_t)(state >> 31), hint_ns, flags);
}

__device__ __noinline__ uint32_t wgemu_mbar_try_wait_parity(uint32_t bar, uint32_t parity,
                                                            uint32_t hint_ns, uint32_t flags) {
  return wgemu::async::try_phase(bar, parity, hint_ns, flags);
}

__device__ __noinline__ uint32_t wgemu_mbar_pending_count(uint64_t state) {
  return (uint32_t)state & (uint32_t)WGEMU_MB_PENDING_MASK;
}

__device__ __noinline__ uint32_t wgemu_mbar_check_layout(uint32_t bar, uint32_t layout_v1) {
  using namespace wgemu::async;
  const uint64_t w = mb_load_acquire(bar);
  return (mb_valid(w) && ((uint32_t)(w >> WGEMU_MB_V1_BIT) & 1u) == (layout_v1 & 1u)) ? 1u : 0u;
}

// cp.async.mbarrier.arrive: all prior cp.async of this thread complete first (cp.async.wait_all),
// then the arrive-on. Without .noinc the pending count is incremented before the asynchronous
// arrive, so the net effect on the current phase is zero and nothing is written.
__device__ __noinline__ void wgemu_cp_async_mbar_arrive(uint32_t bar, uint32_t flags) {
  using namespace wgemu::async;
  asm volatile("cp.async.wait_all;" ::: "memory");
  if (flags & WGEMU_AF_NOINC) {
    mb_update(bar, 1u, 0, false, false, flags);
  } else {
    mb_check_valid(bar, mb_load_relaxed(bar), flags);
  }
}

// ---- TMA ----------------------------------------------------------------------------------------
__device__ __noinline__ void wgemu_tma_load(uint32_t dst, uint64_t tmap, uint32_t mbar, uint32_t c0,
                                            uint32_t c1, uint32_t c2, uint32_t c3, uint32_t c4,
                                            uint32_t o0, uint32_t o1, uint32_t o2,
                                            uint32_t cta_mask, uint32_t flags) {
  using namespace wgemu::async;
  uint32_t bytes;
  if (tma_load_run(dst, tmap, c0, c1, c2, c3, c4, o0, o1, o2, cta_mask, flags, nullptr, bytes)) {
    complete_tx(mbar, bytes, flags);
  }
}

__device__ __noinline__ void wgemu_tma_store(uint64_t tmap, uint32_t src, uint32_t c0, uint32_t c1,
                                             uint32_t c2, uint32_t c3, uint32_t c4, uint32_t flags,
                                             uint64_t bulk_state) {
  using namespace wgemu::async;
  Tmap t;
  tmap_read(tmap, t);
  tensor_check_mode(t, flags);
  if (src & 15u) undefined("tensor-copy-smem-misaligned", src, 0u, flags);
  int32_t c[5];
  coords(c, c0, c1, c2, c3, c4);
  const Xfer x = make_xfer(kStore, t, 0u);
  const uint32_t zero[3] = {0u, 0u, 0u};
  const uint32_t bytes = t.im2col ? im2col_xfer(t, x, src, c, zero) : tile_xfer(t, x, src, c);
  uint32_t lo, n;
  tensor_src_range(src, bytes, t.swizzle, lo, n);
  bulk_record(bulk_state, lo, n);
}

__device__ __noinline__ void wgemu_tma_reduce(uint64_t tmap, uint32_t src, uint32_t c0, uint32_t c1,
                                              uint32_t c2, uint32_t c3, uint32_t c4, uint32_t flags,
                                              uint64_t bulk_state) {
  using namespace wgemu::async;
  Tmap t;
  tmap_read(tmap, t);
  tensor_check_mode(t, flags);
  if (src & 15u) undefined("tensor-copy-smem-misaligned", src, 0u, flags);
  int32_t c[5];
  coords(c, c0, c1, c2, c3, c4);
  const Xfer x = make_xfer(kReduce, t, (flags >> WGEMU_AF_OP_SHIFT) & 0xFu);
  const uint32_t zero[3] = {0u, 0u, 0u};
  const uint32_t bytes = t.im2col ? im2col_xfer(t, x, src, c, zero) : tile_xfer(t, x, src, c);
  uint32_t lo, n;
  tensor_src_range(src, bytes, t.swizzle, lo, n);
  bulk_record(bulk_state, lo, n);
}

// ---- bulk copies --------------------------------------------------------------------------------
__device__ __noinline__ void wgemu_bulk_g2s(uint32_t dst, uint64_t src, uint32_t size,
                                            uint32_t mbar, uint32_t ign_left, uint32_t ign_right,
                                            uint32_t cta_mask, uint32_t flags) {
  using namespace wgemu::async;
  if (!multicast_self(cta_mask, flags)) return;
  bulk_g2s_run(dst, src, size, ign_left, ign_right, flags, nullptr);
  complete_tx(mbar, size, flags);
}

// shared::cta -> shared::cluster. The destination has to be in the shared memory of a different
// CTA of the cluster (ptx.txt 15442-15443); with cluster size 1 no such CTA exists, so this is
// undefined behaviour: reported, then (check mode) the copy is done into the executing CTA.
__device__ __noinline__ void wgemu_bulk_s2s(uint32_t dst, uint32_t src, uint32_t size,
                                            uint32_t mbar, uint32_t flags) {
  using namespace wgemu::async;
  undefined("cp.async.bulk.shared::cluster.shared::cta-needs-remote-cta", dst, src, flags);
  bulk_check_align(dst, src, size, flags);
  uint32_t w[4];
  for (uint32_t o = 0; o < size; o += 16) {
    shared_ld16(src + o, w);
    shared_st16(dst + o, w);
  }
  complete_tx(mbar, size, flags);
}

__device__ __noinline__ void wgemu_bulk_s2g(uint64_t dst, uint32_t src, uint32_t size,
                                            uint32_t flags, uint64_t bulk_state) {
  using namespace wgemu::async;
  bulk_check_align(dst, src, size, flags);
  uint32_t w[4];
  for (uint32_t o = 0; o < size; o += 16) {
    shared_ld16(src + o, w);
    gen_st16(dst + o, w);
  }
  bulk_record(bulk_state, src, size);
}

__device__ __noinline__ void wgemu_bulk_red_s2s(uint32_t dst, uint32_t src, uint32_t size,
                                                uint32_t mbar, uint32_t flags) {
  using namespace wgemu::async;
  bulk_check_align(dst, src, size, flags);
  const uint32_t op = (flags >> WGEMU_AF_OP_SHIFT) & 0xFu;
  const uint32_t ty = (flags >> WGEMU_AF_TYPE_SHIFT) & 0xFu;
  const uint32_t eb = red_bytes(ty);
  for (uint32_t o = 0; o < size; o += eb) {
    red_shared(dst + o, op, ty, shared_ld(src + o, (int)eb));
  }
  complete_tx(mbar, size, flags);
}

__device__ __noinline__ void wgemu_bulk_red_s2g(uint64_t dst, uint32_t src, uint32_t size,
                                                uint32_t flags, uint64_t bulk_state) {
  using namespace wgemu::async;
  bulk_check_align(dst, src, size, flags);
  const uint32_t op = (flags >> WGEMU_AF_OP_SHIFT) & 0xFu;
  const uint32_t ty = (flags >> WGEMU_AF_TYPE_SHIFT) & 0xFu;
  const uint32_t eb = red_bytes(ty);
  for (uint32_t o = 0; o < size; o += eb) {
    red_global(dst + o, op, ty, shared_ld(src + o, (int)eb));
  }
  bulk_record(bulk_state, src, size);
}

__device__ __noinline__ void wgemu_bulk_state_init(uint64_t bulk_state) {
  wgemu::async::gen_st(bulk_state, 8, 0ull);
}

__device__ __noinline__ void wgemu_bulk_commit(uint64_t bulk_state) {
  using namespace wgemu::async;
  gen_st(bulk_state, 4, gen_ld(bulk_state, 4) + 1u);
}

// cp.async.bulk.wait_group{.read} N in check mode: the groups older than the N most recent
// committed ones are complete; their shared-memory sources must still hold the bytes they had at
// issue (a store into a source before this wait is a program error the synchronous copy hides).
__device__ __noinline__ void wgemu_bulk_wait(uint64_t bulk_state, uint32_t n, uint32_t flags) {
  using namespace wgemu::async;
  const uint64_t st = bulk_state;
  const uint32_t cur = (uint32_t)gen_ld(st, 4);
  const uint32_t cnt_raw = (uint32_t)gen_ld(st + 4, 4);
  const uint32_t cnt = cnt_raw & 0xFFFFu;
  uint32_t keep = 0;
  for (uint32_t i = 0; i < cnt; ++i) {
    const uint64_t e = st + 8 + 16ull * i;
    const uint32_t addr = (uint32_t)gen_ld(e, 4);
    const uint32_t bytes = (uint32_t)gen_ld(e + 4, 4);
    const uint32_t hash = (uint32_t)gen_ld(e + 8, 4);
    const uint32_t group = (uint32_t)gen_ld(e + 12, 4);
    if ((uint64_t)group + n < (uint64_t)cur) {
      if (smem_hash(addr, bytes) != hash) {
        report((flags & WGEMU_AF_READ) ? "bulk-source-overwritten-before-wait_group.read"
                                       : "bulk-source-overwritten-before-wait_group",
               addr, bytes, flags);
      }
    } else {
      if (keep != i) {
        const uint64_t d = st + 8 + 16ull * keep;
        gen_st(d, 4, addr);
        gen_st(d + 4, 4, bytes);
        gen_st(d + 8, 4, hash);
        gen_st(d + 12, 4, group);
      }
      ++keep;
    }
  }
  gen_st(st + 4, 4, (cnt_raw & 0x80000000u) | keep);
}

// ---- st.async / red.async -----------------------------------------------------------------------
// Both need %cluster_nctarank > 1 (st.async, ptx.txt 13642-13645) or a target CTA different from
// the executing one (red.async, ptx.txt 20530-20533). Cluster size is 1 here, so every execution
// is undefined behaviour: reported, then (check mode) applied to the executing CTA.
__device__ __noinline__ void wgemu_st_async(uint32_t addr, uint64_t lo, uint64_t hi, uint32_t mbar,
                                            uint32_t flags) {
  using namespace wgemu::async;
  const uint32_t nbytes = flags & WGEMU_AF_NBYTES_MASK;
  undefined("st.async-needs-cluster-nctarank-gt-1", addr, mbar, flags);
  if (addr & (nbytes - 1u)) undefined("st.async-misaligned", addr, nbytes, flags);
  if (nbytes == 4u) {
    shared_st(addr, 4, lo);
  } else if (nbytes == 8u) {
    shared_st(addr, 8, lo);
  } else if (nbytes == 16u) {
    shared_st_u64x2(addr, lo, hi);
  } else {
    fatal("st.async-size", nbytes, addr);
  }
  complete_tx(mbar, nbytes, flags);
}

__device__ __noinline__ void wgemu_red_async(uint32_t addr, uint64_t val, uint32_t mbar,
                                             uint32_t flags) {
  using namespace wgemu::async;
  const uint32_t op = (flags >> WGEMU_AF_OP_SHIFT) & 0xFu;
  const uint32_t ty = (flags >> WGEMU_AF_TYPE_SHIFT) & 0xFu;
  const uint32_t nbytes = red_bytes(ty);
  undefined("red.async-needs-remote-cta", addr, mbar, flags);
  if (addr & (nbytes - 1u)) undefined("red.async-misaligned", addr, nbytes, flags);
  red_shared(addr, op, ty, val);
  complete_tx(mbar, nbytes, flags);
}

// ---- tensor-map objects -------------------------------------------------------------------------
__device__ __noinline__ void wgemu_tmap_replace(uint64_t tmap, uint32_t field, uint32_t ord,
                                                uint64_t val, uint32_t flags) {
  using namespace wgemu::async;
  const uint64_t t = tmap;
  if (t & 7u) fatal("tensormap-misaligned", (uint32_t)t, (uint32_t)(t >> 32));
  const uint32_t magic = (uint32_t)gen_ld(t + 12, 4);
  if (magic != WGEMU_TMAP_MAGIC) fatal("tensormap-not-encoded-by-wgemu-shim", magic, field);
  uint32_t misc = (uint32_t)gen_ld(t + 8, 4);
  const uint32_t v = (uint32_t)val;
  switch (field) {
    case WGEMU_TMR_RANK:
      if (v > 4u) fatal("tensormap.replace-rank", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_RANK_M1_SHIFT, 3, v);
      break;
    case WGEMU_TMR_BOX_DIM:
      if (ord > 4u || v == 0u || v > 256u) fatal("tensormap.replace-box_dim", ord, v);
      gen_st(t + 72 + ord, 1, v - 1u);
      break;
    case WGEMU_TMR_GLOBAL_DIM:
      if (ord > 4u || v == 0u) fatal("tensormap.replace-global_dim", ord, v);
      gen_st(t + 16 + 4 * ord, 4, v - 1u);
      break;
    case WGEMU_TMR_GLOBAL_STRIDE:
      if (ord > 3u) fatal("tensormap.replace-global_stride-ord", ord, 0u);
      gen_st(t + 40 + 8 * ord, 8, val);
      break;
    case WGEMU_TMR_ELEMENT_STRIDE:
      if (ord > 4u || v == 0u || v > 8u) fatal("tensormap.replace-element_stride", ord, v);
      gen_st(t + 80 + ord, 1, v - 1u);
      break;
    case WGEMU_TMR_ELEMTYPE:
      if (v > (uint32_t)WGEMU_TM_TF32_FTZ) fatal("tensormap.replace-elemtype", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_ELEMTYPE_SHIFT, 4, v);
      break;
    case WGEMU_TMR_INTERLEAVE:
      if (v > 2u) fatal("tensormap.replace-interleave_layout", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_INTERLEAVE_SHIFT, 2, v);
      break;
    case WGEMU_TMR_SWIZZLE_MODE:
      if (v > (uint32_t)WGEMU_TM_SWIZZLE_128B) fatal("tensormap.replace-swizzle_mode", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_SWIZZLE_SHIFT, 3, v);
      break;
    case WGEMU_TMR_SWIZZLE_ATOMICITY:
      if (v > 3u) fatal("tensormap.replace-swizzle_atomicity", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_ATOMICITY_SHIFT, 2, v);
      break;
    case WGEMU_TMR_FILL_MODE:
      if (v > 1u) fatal("tensormap.replace-fill_mode", v, 0u);
      misc = wgemu_tm_set(misc, WGEMU_TM_FILL_SHIFT, 1, v);
      break;
    case WGEMU_TMR_GLOBAL_ADDRESS:
      gen_st(t, 8, val);
      break;
    case WGEMU_TMR_ALLOC_CLASS:
      break;
    default:
      fatal("tensormap.replace-field", field, 0u);
  }
  // Allocation class: with .global_address and no class the tensor becomes small (ptx.txt
  // 17066-17068); 3 (.alloc_same) keeps it.
  const uint32_t cls = flags & WGEMU_AF_ALLOC_MASK;
  if (field == WGEMU_TMR_GLOBAL_ADDRESS || field == WGEMU_TMR_ALLOC_CLASS) {
    if (cls == WGEMU_ALLOC_LARGE) {
      misc = wgemu_tm_set(misc, WGEMU_TM_ALLOC_LARGE_SHIFT, 1, 1u);
    } else if (cls != WGEMU_ALLOC_SAME) {
      misc = wgemu_tm_set(misc, WGEMU_TM_ALLOC_LARGE_SHIFT, 1, 0u);
    }
  }
  gen_st(t + 8, 4, misc);
}

// tensormap.cp_fenceproxy copy part: 128 bytes from shared::cta to global. The translator emits
// the release fence after the call.
__device__ __noinline__ void wgemu_tmap_copy(uint64_t dst, uint32_t src, uint32_t size) {
  using namespace wgemu::async;
  if (size != 128u) fatal("tensormap.cp_fenceproxy-size", size, 0u);
  if ((dst & 15u) || (src & 15u)) fatal("tensormap.cp_fenceproxy-misaligned", (uint32_t)dst, src);
  uint32_t w[4];
  for (uint32_t o = 0; o < 128u; o += 16) {
    shared_ld16(src + o, w);
    gen_st16(dst + o, w);
  }
}

__device__ __noinline__ uint32_t wgemu_tmap_query_large(uint64_t tmap) {
  using namespace wgemu::async;
  const uint32_t magic = (uint32_t)gen_ld(tmap + 12, 4);
  if (magic != WGEMU_TMAP_MAGIC) fatal("tensormap-not-encoded-by-wgemu-shim", magic, 0u);
  return wgemu_tm_get((uint32_t)gen_ld(tmap + 8, 4), WGEMU_TM_ALLOC_LARGE_SHIFT, 1);
}

#ifndef WGEMU_SMEM_GLOBAL  // E1 delayed delivery is not part of the E4 build
// ---- E1 delayed delivery ------------------------------------------------------------------------
// Entry prologue of every kernel, all threads. Thread 0 claims a pool slot and writes the header.
__device__ __noinline__ void wgemu_dl_init(uint32_t hdr, uint64_t pool, uint32_t nslots,
                                           uint32_t nrec, uint32_t flags) {
  using namespace wgemu::async;
  const uint32_t tid = linear_tid();
  if (tid == 0u) {
    const uint32_t nthr = blockDim.x * blockDim.y * blockDim.z;
    if (nthr > WGEMU_DL_MAX_THREADS) fatal("async-delay-block-size", nthr, WGEMU_DL_MAX_THREADS);
    if (nslots == 0u || nrec == 0u || nrec > 0xFFFEu) fatal("async-delay-pool-shape", nslots, nrec);
    const uint64_t sb = WGEMU_DL_SLOT_BYTES((uint64_t)nrec);
    const uint32_t start =
        (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) % nslots;
    const uint64_t t0 = globaltimer();
    uint64_t slot = 0ull;
    for (uint32_t k = 0;; ++k) {
      const uint64_t cand = pool + sb * ((start + k) % nslots);
      if (atomicCAS(reinterpret_cast<unsigned int*>(cand), 0u, 1u) == 0u) {
        __threadfence();
        slot = cand;
        break;
      }
      if ((k + 1u) % nslots == 0u) {
        if (globaltimer() - t0 > kDlSlotWaitNs) {
          report("async-delay-no-slot", nslots, nrec, flags);
          break;
        }
        nanosleep_ns(1000);
      }
    }
    shared_st(hdr + kHdrSlot, 8, slot);
    hdr_st(hdr, kHdrLock, 0u);
    hdr_st(hdr, kHdrAlive, nthr);
    hdr_st(hdr, kHdrLive, 0u);
    hdr_st(hdr, kHdrHiwater, 0u);
    hdr_st(hdr, kHdrNrec, nrec);
    hdr_st(hdr, kHdrMagic, WGEMU_DL_MAGIC);
  }
  __syncthreads();
  const uint64_t slot = shared_ld(hdr + kHdrSlot, 8);
  if (slot != 0ull) {
    gen_st(dl_tword(slot, WGEMU_DL_SLOT_TGROUP, tid), 4, 0ull);
    gen_st(dl_tword(slot, WGEMU_DL_SLOT_TCOUNT, tid), 4, 0ull);
  }
}

// Before every exit and every ret of an entry. The last thread of the CTA drops the copies still
// pending (their destinations keep the poison) and releases the slot.
__device__ __noinline__ void wgemu_dl_exit(uint32_t hdr, uint32_t flags) {
  using namespace wgemu::async;
  (void)flags;
  if (hdr_ld(hdr, kHdrMagic) != WGEMU_DL_MAGIC) return;
  if (sh_atom_add(hdr + kHdrAlive, 0xFFFFFFFFu) != 1u) return;
  const uint64_t slot = shared_ld(hdr + kHdrSlot, 8);
  if (slot == 0ull) return;
  dl_lock(hdr);
  const uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  const uint64_t owner = slot + WGEMU_DL_SLOT_OWNER;
  for (uint32_t i = 0; i < hw; ++i) {
    const uint64_t r = dl_rec(slot, i);
    if ((uint32_t)gen_ld(r + kRecInfo, 4) & WGEMU_DL_LIVE) {
      const uint32_t lo = (uint32_t)gen_ld(r + kRecLo, 4);
      const uint32_t span = (uint32_t)gen_ld(r + kRecSpan, 4);
      for (uint32_t a = lo; a < lo + span; a += 4u) {
        const uint64_t o = owner + 2ull * (a >> 2);
        if ((uint32_t)gen_ld(o, 2) == i + 1u) gen_st(o, 2, 0ull);
      }
    }
    gen_st(r + kRecInfo, 4, 0ull);
  }
  hdr_st(hdr, kHdrLive, 0u);
  hdr_st(hdr, kHdrHiwater, 0u);
  shared_st(hdr + kHdrSlot, 8, 0ull);
  dl_unlock(hdr);
  __threadfence();
  atomicExch(reinterpret_cast<unsigned int*>(slot), 0u);
}

__device__ __noinline__ void wgemu_dl_tma_load(uint32_t dst, uint64_t tmap, uint32_t mbar,
                                               uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                               uint32_t c4, uint32_t o0, uint32_t o1, uint32_t o2,
                                               uint32_t cta_mask, uint32_t flags, uint32_t hdr) {
  using namespace wgemu::async;
  const uint64_t slot = dl_slot(hdr);
  if (slot == 0ull) {
    wgemu_tma_load(dst, tmap, mbar, c0, c1, c2, c3, c4, o0, o1, o2, cta_mask, flags);
    return;
  }
  dl_lock(hdr);
  const uint32_t idx = dl_alloc(hdr, slot, flags);
  Sink s;
  if (idx != kDlNone) dl_sink(hdr, slot, idx, s);
  uint32_t bytes;
  if (tma_load_run(dst, tmap, c0, c1, c2, c3, c4, o0, o1, o2, cta_mask, flags,
                   idx != kDlNone ? &s : nullptr, bytes)) {
    const uint64_t old = complete_tx(mbar, bytes, flags);
    if (idx != kDlNone) {
      const uint32_t par = mb_phase(old) ? WGEMU_DL_PARITY : 0u;
      dl_commit(hdr, slot, idx, s, mbar, WGEMU_DL_BOUND | par, 0u, flags);
    }
  }
  dl_unlock(hdr);
}

__device__ __noinline__ void wgemu_dl_bulk_g2s(uint32_t dst, uint64_t src, uint32_t size,
                                               uint32_t mbar, uint32_t ign_left, uint32_t ign_right,
                                               uint32_t cta_mask, uint32_t flags, uint32_t hdr) {
  using namespace wgemu::async;
  const uint64_t slot = dl_slot(hdr);
  if (slot == 0ull) {
    wgemu_bulk_g2s(dst, src, size, mbar, ign_left, ign_right, cta_mask, flags);
    return;
  }
  if (!multicast_self(cta_mask, flags)) return;
  dl_lock(hdr);
  const uint32_t idx = dl_alloc(hdr, slot, flags);
  Sink s;
  if (idx != kDlNone) dl_sink(hdr, slot, idx, s);
  bulk_g2s_run(dst, src, size, ign_left, ign_right, flags, idx != kDlNone ? &s : nullptr);
  const uint64_t old = complete_tx(mbar, size, flags);
  if (idx != kDlNone) {
    const uint32_t par = mb_phase(old) ? WGEMU_DL_PARITY : 0u;
    dl_commit(hdr, slot, idx, s, mbar, WGEMU_DL_BOUND | par, 0u, flags);
  }
  dl_unlock(hdr);
}

// After every mbarrier.test_wait / try_wait, by every thread that executed it: `ok` is the wait
// result, `parity` the phase parity waited for. One lane per group of lanes with the same
// (bar, parity, ok) delivers; the others wait for it at __syncwarp.
__device__ __noinline__ void wgemu_dl_observe(uint32_t hdr, uint32_t bar, uint32_t parity,
                                              uint32_t ok, uint32_t flags) {
  using namespace wgemu::async;
  const uint32_t m = __activemask();
  const uint64_t key = ((uint64_t)bar << 2) | ((parity & 1u) << 1) | (ok ? 1u : 0u);
  const uint32_t g = __match_any_sync(m, key);
  if (ok && warp_lane() == (uint32_t)(__ffs(g) - 1)) dl_deliver_phase(hdr, bar, parity, flags);
  __syncwarp(g);
}

// Non-bulk cp.async: src_size bytes from global memory, zeros up to cp_size.
__device__ __noinline__ void wgemu_dl_cp_async(uint32_t hdr, uint32_t dst, uint64_t src,
                                               uint32_t cp_size, uint32_t src_size,
                                               uint32_t flags) {
  using namespace wgemu::async;
  if ((cp_size != 4u && cp_size != 8u && cp_size != 16u) || (dst & (cp_size - 1u)) ||
      (src & (cp_size - 1u))) {
    undefined("cp-async-size-or-alignment", dst, cp_size, flags);
  }
  if (src_size > cp_size) {
    undefined("cp-async-src-size", src_size, cp_size, flags);
    src_size = cp_size;
  }
  const uint64_t slot = dl_slot(hdr);
  Sink s;
  uint32_t idx = kDlNone;
  if (slot != 0ull) {
    dl_lock(hdr);
    idx = dl_alloc(hdr, slot, flags);
    if (idx != kDlNone) dl_sink(hdr, slot, idx, s);
  }
  Sink* sink = idx != kDlNone ? &s : nullptr;
  if (cp_size == 16u && src_size == 16u) {
    uint32_t w[4];
    gen_ld16(src, w);
    dst_st16(sink, dst, w);
  } else {
    for (uint32_t o = 0; o < cp_size; o += 4u) {
      uint32_t v = 0u;
      for (uint32_t b = 0; b < 4u; ++b) {
        if (o + b < src_size) v |= (uint32_t)gen_ld(src + o + b, 1) << (8u * b);
      }
      dst_st(sink, dst + o, 4, v);
    }
  }
  if (slot != 0ull) {
    if (idx != kDlNone) {
      const uint32_t tid = linear_tid();
      const uint64_t tc = dl_tword(slot, WGEMU_DL_SLOT_TCOUNT, tid);
      const uint32_t grp = (uint32_t)gen_ld(dl_tword(slot, WGEMU_DL_SLOT_TGROUP, tid), 4);
      dl_commit(hdr, slot, idx, s, 0u, WGEMU_DL_CP | (tid << WGEMU_DL_OWNER_SHIFT), grp, flags);
      gen_st(tc, 4, gen_ld(tc, 4) + 1u);
    }
    dl_unlock(hdr);
  }
}

__device__ __noinline__ void wgemu_dl_cp_commit(uint32_t hdr) {
  using namespace wgemu::async;
  const uint64_t slot = dl_slot(hdr);
  if (slot == 0ull) return;
  const uint64_t g = dl_tword(slot, WGEMU_DL_SLOT_TGROUP, linear_tid());
  gen_st(g, 4, gen_ld(g, 4) + 1u);
}

// cp.async.wait_group n: the copies of this thread's committed groups except the newest n arrive.
__device__ __noinline__ void wgemu_dl_cp_wait(uint32_t hdr, uint32_t n, uint32_t flags) {
  using namespace wgemu::async;
  const uint64_t slot = dl_slot(hdr);
  if (slot == 0ull) return;
  const uint32_t tid = linear_tid();
  const uint64_t gw = dl_tword(slot, WGEMU_DL_SLOT_TGROUP, tid);
  if (flags & WGEMU_AF_DL_ALL) {
    gen_st(gw, 4, gen_ld(gw, 4) + 1u);
    n = 0u;
  }
  if ((uint32_t)gen_ld(dl_tword(slot, WGEMU_DL_SLOT_TCOUNT, tid), 4) == 0u) return;
  dl_lock(hdr);
  const uint32_t cur = (uint32_t)gen_ld(gw, 4);
  const uint32_t mine = WGEMU_DL_LIVE | WGEMU_DL_CP | (tid << WGEMU_DL_OWNER_SHIFT);
  const uint32_t mask = WGEMU_DL_LIVE | WGEMU_DL_CP | (0xFFFFFFFFu << WGEMU_DL_OWNER_SHIFT);
  const uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  for (uint32_t i = 0; i < hw; ++i) {
    const uint64_t r = dl_rec(slot, i);
    if (((uint32_t)gen_ld(r + kRecInfo, 4) & mask) == mine &&
        cur - (uint32_t)gen_ld(r + kRecGroup, 4) > n) {
      dl_deliver(hdr, slot, i, flags);
    }
  }
  dl_shrink(hdr, slot);
  dl_unlock(hdr);
}

// cp.async.mbarrier.arrive(.noinc): this thread's pending cp.async copies not yet tied to a
// barrier arrive when a thread observes the phase this arrive belongs to.
__device__ __noinline__ void wgemu_dl_cp_mbar_arrive(uint32_t hdr, uint32_t bar, uint32_t flags) {
  using namespace wgemu::async;
  const uint64_t slot = dl_slot(hdr);
  if (slot != 0ull) dl_lock(hdr);
  uint64_t old;
  if (flags & WGEMU_AF_NOINC) {
    old = mb_update(bar, 1u, 0, false, false, flags);
  } else {
    old = mb_load_relaxed(bar);
    mb_check_valid(bar, old, flags);
  }
  if (slot == 0ull) return;
  const uint32_t tid = linear_tid();
  const uint32_t par = mb_phase(old) ? WGEMU_DL_PARITY : 0u;
  const uint32_t mine = WGEMU_DL_LIVE | WGEMU_DL_CP | (tid << WGEMU_DL_OWNER_SHIFT);
  const uint32_t mask =
      WGEMU_DL_LIVE | WGEMU_DL_CP | WGEMU_DL_BOUND | (0xFFFFFFFFu << WGEMU_DL_OWNER_SHIFT);
  const uint32_t hw = hdr_ld(hdr, kHdrHiwater);
  for (uint32_t i = 0; i < hw; ++i) {
    const uint64_t r = dl_rec(slot, i);
    const uint32_t info = (uint32_t)gen_ld(r + kRecInfo, 4);
    if ((info & mask) == mine) {
      gen_st(r + kRecBar, 4, bar);
      gen_st(r + kRecInfo, 4, info | WGEMU_DL_BOUND | par);
    }
  }
  dl_unlock(hdr);
}

#endif  // !WGEMU_SMEM_GLOBAL

// ---- E2 counters --------------------------------------------------------------------------------
// counter[off] += sum of v over the active lanes that add to the same counter (one atomic per
// counter). Diverged lanes can enter together with different offsets, e.g. a bulk copy issuer
// next to lanes leaving an mbarrier wait.
__device__ __noinline__ void wgemu_count_add(uint64_t cnt, uint32_t off, uint32_t v) {
  using namespace wgemu::async;
  const uint32_t g = __match_any_sync(__activemask(), off);
  const uint32_t sum = __reduce_add_sync(g, v);
  if (sum != 0u && warp_lane() == (uint32_t)(__ffs(g) - 1)) {
    atomicAdd(reinterpret_cast<unsigned long long*>(cnt + off), (unsigned long long)sum);
  }
}

// One count per warpgroup (its first thread) in the table of wgmma codes.
__device__ __noinline__ void wgemu_count_wgmma(uint64_t cnt, uint32_t code) {
  using namespace wgemu::async;
  if ((linear_tid() & 127u) != 0u) return;
  uint32_t h = (code * 2654435761u) >> 26;  // 6 bits: WGEMU_CNT_WGMMA_SLOTS == 64
  for (uint32_t k = 0; k < WGEMU_CNT_WGMMA_SLOTS; ++k) {
    const uint64_t e = cnt + WGEMU_CNT_WGMMA_TABLE + 16ull * ((h + k) % WGEMU_CNT_WGMMA_SLOTS);
    const uint32_t cur = atomicCAS(reinterpret_cast<unsigned int*>(e), 0u, code);
    if (cur == 0u || cur == code) {
      atomicAdd(reinterpret_cast<unsigned long long*>(e + 8), 1ull);
      return;
    }
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(cnt + WGEMU_CNT_WGMMA_OVERFLOW), 1ull);
}

__device__ __noinline__ uint32_t wgemu_tma_bytes(uint64_t tmap, uint32_t flags) {
  using namespace wgemu::async;
  (void)flags;
  Tmap t;
  tmap_read(tmap, t);
  return tensor_box_bytes(t);
}

#ifdef WGEMU_SMEM_GLOBAL
// ---- E4 shadow shared memory --------------------------------------------------------------------
// The CTA's shared window is a slot of the global pool, addressed by virtual offsets (see
// include/wgmma_emu/e4_device.cuh). A slot is leased with atom.acquire.gpu.cas on its lease word
// and returned with atom.release.gpu.exch by the last thread of the CTA, so the shadow bytes of a
// finished CTA are visible to the next CTA that leases the slot. The lease and stats addresses
// live in the slot bytes below WGEMU_E4_ORIGIN (no shared access can reach them).
constexpr uint64_t kE4LeaseWaitNs = 10000000000ull;  // 10 s

// Prologue of every E4 entry, all threads (meta = (WGEMU_E4_MAGIC << 32) | dynamic bytes).
__device__ __noinline__ void wgemu_e4_enter(uint64_t meta, uint32_t dyn_base, uint64_t pool,
                                            uint64_t lease, uint64_t stats, uint32_t slots) {
  using namespace wgemu::async;
  if (linear_tid() == 0u) {
    if ((uint32_t)(meta >> 32) != WGEMU_E4_MAGIC) {
      fatal("e4-meta: launched without the shim's metadata", (uint32_t)(meta >> 32),
            (uint32_t)meta);
    }
    uint32_t dsz;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz != 0u) fatal("e4-dynamic-smem-nonzero", dsz, 0u);
    const uint64_t window64 = (uint64_t)dyn_base + (uint64_t)(uint32_t)meta;
    if (window64 > WGEMU_E4_WINDOW_MAX || window64 > WGEMU_E4_STRIDE ||
        window64 < WGEMU_E4_ORIGIN || window64 - WGEMU_E4_ORIGIN > WGEMU_E4_SM90_BLOCK_MAX) {
      fatal("e4-window", (uint32_t)window64, dyn_base);
    }
    if (slots == 0u) fatal("e4-slots", slots, 0u);
    const uint32_t window = (uint32_t)window64;
    const uint32_t nthr = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const uint64_t t0 = globaltimer();
    const uint32_t h = (lin * 2654435761u) ^ (smid * 40503u) ^ (uint32_t)(t0 >> 7) * 2246822519u;
    uint32_t idx = 0xFFFFFFFFu;
    for (;;) {
      for (uint32_t k = 0; k < slots; ++k) {
        const uint32_t i = (h + k) % slots;
        uint32_t old;
        asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], 0, 1;"
                     : "=r"(old)
                     : "l"(to_global(lease + 4ull * i))
                     : "memory");
        if (old == 0u) {
          idx = i;
          break;
        }
      }
      if (idx != 0xFFFFFFFFu) break;
      if (globaltimer() - t0 > kE4LeaseWaitNs) {
        fatal("e4-no-slot: raise WGEMU_E4_SLOTS (slots, grid CTAs)", slots,
              gridDim.x * gridDim.y * gridDim.z);
      }
      nanosleep_ns(1000);
    }
    const uint64_t slot = pool + (uint64_t)WGEMU_E4_STRIDE * idx;
    gen_st(slot + WGEMU_E4_S_LEASE, 8, lease + 4ull * idx);
    gen_st(slot + WGEMU_E4_S_STATS, 8, stats);
    atomicAdd(reinterpret_cast<unsigned long long*>(stats), 1ull);
    atomicMax(reinterpret_cast<unsigned long long*>(stats + 8), (unsigned long long)window);
    const unsigned long long cur =
        atomicAdd(reinterpret_cast<unsigned long long*>(stats + 24), 1ull) + 1ull;
    atomicMax(reinterpret_cast<unsigned long long*>(stats + 16), cur);
    wgemu::e4::hdr_st64<WGEMU_E4_H_BASE>(slot);
    wgemu::e4::hdr_st32<WGEMU_E4_H_WINDOW>(window);
    wgemu::e4::hdr_st32<WGEMU_E4_H_DYN>((uint32_t)meta);
    wgemu::e4::hdr_st32<WGEMU_E4_H_ALIVE>(nthr);
    wgemu::e4::hdr_st32<WGEMU_E4_H_SLOT>(idx);
    wgemu::e4::hdr_st32<WGEMU_E4_H_MAGIC>(WGEMU_E4_MAGIC);
  }
  __syncthreads();
}

// Before every ret and exit of an E4 entry, all threads. The last thread of the CTA clears the
// header and returns the slot.
__device__ __noinline__ void wgemu_e4_exit() {
  using namespace wgemu::async;
  using namespace wgemu::e4;
  if (hdr_ld32<WGEMU_E4_H_MAGIC>() != WGEMU_E4_MAGIC) return;
  const uint64_t slot = hdr_ld64<WGEMU_E4_H_BASE>();
  if (hdr_alive_dec() != 1u) return;
  const uint64_t lease = gen_ld(slot + WGEMU_E4_S_LEASE, 8);
  const uint64_t stats = gen_ld(slot + WGEMU_E4_S_STATS, 8);
  hdr_st32<WGEMU_E4_H_MAGIC>(0u);
  hdr_st64<WGEMU_E4_H_BASE>(0ull);
  hdr_st32<WGEMU_E4_H_WINDOW>(0u);
  atomicAdd(reinterpret_cast<unsigned long long*>(stats + 24), 0xFFFFFFFFFFFFFFFFull);
  uint32_t old;
  asm volatile("atom.release.gpu.global.exch.b32 %0, [%1], 0;"
               : "=r"(old)
               : "l"(to_global(lease))
               : "memory");
  asm volatile("" ::"r"(old));  // the exchange result is not needed
}

// Non-bulk cp.async (SM80 form): cp_size 4, 8 or 16 bytes, src_size bytes copied, zero fill
// after them. The copy completes before the call returns.
__device__ __noinline__ void wgemu_e4_cp_async(uint32_t dst, uint64_t src, uint32_t cp_size,
                                               uint32_t src_size) {
  using namespace wgemu::async;
  if ((cp_size != 4u && cp_size != 8u && cp_size != 16u) || (dst & (cp_size - 1u)) ||
      (src_size != 0u && (src & (cp_size - 1u)))) {
    fatal("e4-cp-async-size-or-alignment", dst, cp_size);
  }
  if (src_size > cp_size) fatal("e4-cp-async-src-size", src_size, cp_size);
  if (cp_size == 16u && src_size == 16u) {
    uint32_t w[4];
    gen_ld16(src, w);
    shared_st16(dst, w);
    return;
  }
  for (uint32_t o = 0; o < cp_size; o += 4u) {
    uint32_t v = 0u;
    if (o + 4u <= src_size) {
      v = (uint32_t)gen_ld(src + o, 4);
    } else {
      for (uint32_t b = 0; b < 4u; ++b) {
        if (o + b < src_size) v |= (uint32_t)gen_ld(src + o + b, 1) << (8u * b);
      }
    }
    shared_st(dst + o, 4, v);
  }
}

// cp.async.mbarrier.arrive(.noinc): the copies of this thread are complete already. With .noinc
// the arrive-on counts once (release: the shadow writes above are ordered before it); without it
// the pending count is incremented and decremented again, so only the barrier address is checked.
__device__ __noinline__ void wgemu_e4_cp_async_arrive(uint32_t bar, uint32_t noinc) {
  using namespace wgemu::async;
  if (bar & 7u) fatal("mbarrier-misaligned", bar, 0u);
  if (noinc) {
    mb_update(bar, 1u, 0, false, false, 0u);
  } else {
    mb_check_valid(bar, mb_load_relaxed(bar), 0u);
  }
}

__device__ __noinline__ uint32_t wgemu_e4_dyn_smem() {
  using namespace wgemu::e4;
  if (hdr_ld32<WGEMU_E4_H_MAGIC>() != WGEMU_E4_MAGIC) {
    wgemu::async::fatal("e4-header-uninitialized", 0u, 0u);
  }
  return hdr_ld32<WGEMU_E4_H_DYN>();
}

__device__ __noinline__ uint32_t wgemu_e4_total_smem() {
  using namespace wgemu::e4;
  if (hdr_ld32<WGEMU_E4_H_MAGIC>() != WGEMU_E4_MAGIC) {
    wgemu::async::fatal("e4-header-uninitialized", 0u, 0u);
  }
  return hdr_ld32<WGEMU_E4_H_WINDOW>() - WGEMU_E4_ORIGIN;
}
#endif  // WGEMU_SMEM_GLOBAL

}  // extern "C"
