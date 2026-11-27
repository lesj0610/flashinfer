// Asynchronous-copy, TMA and mbarrier emulation for the SM80 wgmma emulation layer (fork-only).
// Owner: A. Implementation: runtime/lib_async.cu (included by runtime/emu_lib.cu). Translator:
// tools/ptxemu/rules_async.py. Tensor-map format: include/wgmma_emu/tensormap.h.
//
// Library entry points (PLAN.md section 6: extern "C" __device__ __noinline__, u32/u64 scalars,
// pointers are generic u64, shared-memory addresses are 32-bit shared-window u32, at most one u32
// return value). Bit 31 of every `flags` argument is WGEMU_AF_CHECK (translation in check mode),
// bit 30 is WGEMU_AF_TRAP (WGEMU_TRAP=1: a reported violation also traps).
//
//   mbarrier (64-bit word in our format, see "mbarrier word" below)
//     void     wgemu_mbar_init(u32 bar, u32 count, u32 flags)           flags: WGEMU_AF_MB_V1
//     void     wgemu_mbar_inval(u32 bar, u32 flags)
//     u32      wgemu_mbar_arrive(u32 bar, u32 count, u32 tx, u32 flags) returns state (low word)
//              flags: WGEMU_AF_MB_NOCOMPLETE, WGEMU_AF_MB_DROP, WGEMU_AF_MB_EXPECT_TX
//     void     wgemu_mbar_expect_tx(u32 bar, u32 tx, u32 flags)
//     void     wgemu_mbar_complete_tx(u32 bar, u32 tx, u32 flags)
//     u32      wgemu_mbar_test_wait(u32 bar, u64 state, u32 flags)                 1 = complete
//     u32      wgemu_mbar_test_wait_parity(u32 bar, u32 parity, u32 flags)
//     u32      wgemu_mbar_try_wait(u32 bar, u64 state, u32 hint_ns, u32 flags)    flags: _HINT
//     u32      wgemu_mbar_try_wait_parity(u32 bar, u32 parity, u32 hint_ns, u32 flags)
//     u32      wgemu_mbar_pending_count(u64 state)
//     u32      wgemu_mbar_check_layout(u32 bar, u32 layout_v1)
//     void     wgemu_cp_async_mbar_arrive(u32 bar, u32 flags)                     flags: _NOINC
//   TMA (tensor copies; the issuing thread copies synchronously, then complete-tx)
//     void     wgemu_tma_load(u32 dst, u64 tmap, u32 mbar, u32 c0, u32 c1, u32 c2, u32 c3,
//                             u32 c4, u32 o0, u32 o1, u32 o2, u32 cta_mask, u32 flags)
//              flags: dim (bits 0-2), WGEMU_AF_IM2COL, WGEMU_AF_MULTICAST
//     void     wgemu_tma_store(u64 tmap, u32 src, u32 c0, u32 c1, u32 c2, u32 c3, u32 c4,
//                              u32 flags, u64 bulk_state)        flags: dim, WGEMU_AF_IM2COL
//     void     wgemu_tma_reduce(u64 tmap, u32 src, u32 c0, u32 c1, u32 c2, u32 c3, u32 c4,
//                               u32 flags, u64 bulk_state)       flags: dim, _IM2COL, op (8-11)
//   bulk copies and reductions (non-tensor)
//     void     wgemu_bulk_g2s(u32 dst, u64 src, u32 size, u32 mbar, u32 ign_left,
//                             u32 ign_right, u32 cta_mask, u32 flags)        flags: _MULTICAST
//     void     wgemu_bulk_s2s(u32 dst, u32 src, u32 size, u32 mbar, u32 flags)
//     void     wgemu_bulk_s2g(u64 dst, u32 src, u32 size, u32 flags, u64 bulk_state)
//     void     wgemu_bulk_red_s2s(u32 dst, u32 src, u32 size, u32 mbar, u32 flags)
//     void     wgemu_bulk_red_s2g(u64 dst, u32 src, u32 size, u32 flags, u64 bulk_state)
//              reduction flags: op (bits 8-11), type (bits 12-15)
//     void     wgemu_bulk_state_init(u64 bulk_state)                 check mode only
//     void     wgemu_bulk_commit(u64 bulk_state)                     check mode only
//     void     wgemu_bulk_wait(u64 bulk_state, u32 n, u32 flags)     check mode only; _READ
//   st.async / red.async (they need a remote CTA, %cluster_nctarank > 1; with cluster size 1 the
//   call reports undefined behaviour, then stores into the executing CTA)
//     void     wgemu_st_async(u32 addr, u64 lo, u64 hi, u32 mbar, u32 flags)  flags: bytes 4/8/16
//     void     wgemu_red_async(u32 addr, u64 val, u32 mbar, u32 flags)           op, type
//   tensor-map objects
//     void     wgemu_tmap_replace(u64 tmap, u32 field, u32 ord, u64 val, u32 flags)
//              flags bits 0-1: alloc class (0 unspecified, 1 small, 2 large, 3 same)
//     void     wgemu_tmap_copy(u64 dst, u32 src_smem, u32 size)      tensormap.cp_fenceproxy copy
//     u32      wgemu_tmap_query_large(u64 tmap)
//   E1 delayed delivery (translator --async-delay, nvcc_emu WGEMU_ASYNC_DELAY=1; `hdr` is the
//   32-bit shared address of the per-CTA header, see "E1 delayed delivery" below)
//     void     wgemu_dl_init(u32 hdr, u64 pool, u32 nslots, u32 nrec, u32 flags)   every entry
//     void     wgemu_dl_exit(u32 hdr, u32 flags)                  before ret (entry) and exit
//     void     wgemu_dl_tma_load(<wgemu_tma_load arguments>, u32 hdr)
//     void     wgemu_dl_bulk_g2s(<wgemu_bulk_g2s arguments>, u32 hdr)
//     void     wgemu_dl_observe(u32 hdr, u32 bar, u32 parity, u32 ok, u32 flags)
//              after every mbarrier.test_wait / try_wait (ok = its result, all lanes call it)
//     void     wgemu_dl_cp_async(u32 hdr, u32 dst, u64 src, u32 cp_size, u32 src_size,
//                                u32 flags)                       non-bulk cp.async
//     void     wgemu_dl_cp_commit(u32 hdr)                        cp.async.commit_group
//     void     wgemu_dl_cp_wait(u32 hdr, u32 n, u32 flags)        cp.async.wait_group n /
//              wait_all (flags WGEMU_AF_DL_ALL: commit first, then n = 0)
//     void     wgemu_dl_cp_mbar_arrive(u32 hdr, u32 bar, u32 flags)   flags: _NOINC
//   E2 counters (translator --count, nvcc_emu WGEMU_COUNT=1; `cnt` is the generic address of the
//   module counter block __wgemu_global_count_v1, see "E2 counters" below)
//     void     wgemu_count_wgmma(u64 cnt, u32 code)               counts once per warpgroup
//     u32      wgemu_tma_bytes(u64 tmap, u32 flags)               box bytes of one tensor copy
//
// No call is emitted for prefetch.tensormap, cp.async.bulk.prefetch(.tensor) (hints without
// memory effect), and for cp.async.bulk.commit_group / wait_group outside check mode (every bulk
// operation has completed when its call returns). Fences are rewritten inline by the translator.
//
// mbarrier word (u64, ptx.txt Table 43 ranges):
//   bits  0-19 pending arrival count      bits 20-39 expected arrival count
//   bits 40-60 tx-count, 21-bit two's complement, range -(2^20-1)..2^20-1
//   bit  61    phase parity               bit  62    initialized (check mode uses it)
//   bit  63    .layout::v1
// Updates use atom.acq_rel.cta.shared::cta.cas.b64 (cluster size 1: cta scope covers the cluster).
// Arrive state returned to the program: bit 31 = phase parity before the arrive, bits 0-19 =
// pending count before the arrive (mbarrier.pending_count), zero-extended to the b64 register.
#pragma once

#include <stdint.h>

#include "e4_device.cuh"  // E4 shadow shared memory mode (empty unless WGEMU_SMEM_GLOBAL)
#include "tensormap.h"
#include "types.h"

// ---- flags shared with tools/ptxemu/rules_async.py (keep the values in sync) -------------------
#define WGEMU_AF_CHECK 0x80000000u
#define WGEMU_AF_TRAP 0x40000000u
#define WGEMU_AF_DIM_MASK 0x7u
#define WGEMU_AF_IM2COL 0x8u
#define WGEMU_AF_MULTICAST 0x10u
#define WGEMU_AF_MB_V1 0x1u
#define WGEMU_AF_MB_NOCOMPLETE 0x2u
#define WGEMU_AF_MB_DROP 0x4u
#define WGEMU_AF_MB_EXPECT_TX 0x8u
#define WGEMU_AF_MB_HINT 0x10u
#define WGEMU_AF_NOINC 0x2u
#define WGEMU_AF_READ 0x1u
#define WGEMU_AF_OP_SHIFT 8
#define WGEMU_AF_TYPE_SHIFT 12
#define WGEMU_AF_NBYTES_MASK 0x1Fu
#define WGEMU_AF_ALLOC_MASK 0x3u

// tensormap.replace allocation class (flags bits 0-1).
#define WGEMU_ALLOC_UNSPECIFIED 0u
#define WGEMU_ALLOC_SMALL 1u
#define WGEMU_ALLOC_LARGE 2u
#define WGEMU_ALLOC_SAME 3u

// Reduction operations (.redOp) and element types for bulk / async reductions.
#define WGEMU_RED_ADD 0
#define WGEMU_RED_MIN 1
#define WGEMU_RED_MAX 2
#define WGEMU_RED_INC 3
#define WGEMU_RED_DEC 4
#define WGEMU_RED_AND 5
#define WGEMU_RED_OR 6
#define WGEMU_RED_XOR 7

#define WGEMU_RT_B32 0
#define WGEMU_RT_U32 1
#define WGEMU_RT_S32 2
#define WGEMU_RT_B64 3
#define WGEMU_RT_U64 4
#define WGEMU_RT_S64 5
#define WGEMU_RT_F16 6
#define WGEMU_RT_BF16 7
#define WGEMU_RT_F32 8
#define WGEMU_RT_F64 9
#define WGEMU_RT_F32_FTZ 10

// tensormap.replace fields.
#define WGEMU_TMR_RANK 0
#define WGEMU_TMR_BOX_DIM 1
#define WGEMU_TMR_GLOBAL_DIM 2
#define WGEMU_TMR_GLOBAL_STRIDE 3
#define WGEMU_TMR_ELEMENT_STRIDE 4
#define WGEMU_TMR_ELEMTYPE 5
#define WGEMU_TMR_INTERLEAVE 6
#define WGEMU_TMR_SWIZZLE_MODE 7
#define WGEMU_TMR_SWIZZLE_ATOMICITY 8
#define WGEMU_TMR_FILL_MODE 9
#define WGEMU_TMR_GLOBAL_ADDRESS 10
#define WGEMU_TMR_ALLOC_CLASS 11

// Check-mode bulk-group table (per function, in .local memory, generic address passed in).
#define WGEMU_BULK_MAX 32
#define WGEMU_BULK_STATE_BYTES (8 + 16 * WGEMU_BULK_MAX)

// ---- E1 delayed delivery (keep in sync with tools/ptxemu/instrument.py) ------------------
// An asynchronous copy into shared memory (cp.async.bulk global->shared, cp.async.bulk.tensor
// loads, non-bulk cp.async) writes its data into a per-CTA shadow buffer in global memory and
// WGEMU_DL_POISON bytes into the destination when it is issued. The data reaches shared memory
// when a thread observes the completion: a successful mbarrier test_wait / try_wait on the
// barrier and phase parity the copy completes (also for cp.async.mbarrier.arrive), or, for
// non-bulk cp.async, the issuing thread's cp.async.wait_group / wait_all.
//
// Per-CTA header (WGEMU_DL_HDR_BYTES of module .shared, written by wgemu_dl_init):
//   +0 u64 slot (generic address of the CTA's pool slot; 0: no slot, copies are immediate)
//   +8 u32 lock, +12 u32 threads not exited, +16 u32 live records, +20 u32 record high-water
//   mark, +24 u32 record capacity, +28 u32 WGEMU_DL_MAGIC
// Pool: .weak .global __wgemu_global_async_delay_pool of nslots * WGEMU_DL_SLOT_BYTES(nrec).
// Slot layout (byte offsets):
//   0      u32 claimed (0 free, 1 used by a running CTA)
//   64     u32 cp.async committed-group count per thread (WGEMU_DL_MAX_THREADS)
//   4160   u32 live cp.async records per thread
//   8256   u16 owner per 4-byte shared word (record index + 1, 0: none)
//   WGEMU_DL_SLOT_RECORDS   nrec records of WGEMU_DL_REC_BYTES:
//          u32 lo, u32 span (bytes, whole 4-byte words), u32 bar, u32 info, u32 group, 3 x pad
//          info: WGEMU_DL_LIVE, _BOUND, _PARITY, _CP, owner thread at WGEMU_DL_OWNER_SHIFT
//   WGEMU_DL_SLOT_RECORDS + nrec * WGEMU_DL_REC_BYTES   shadow, indexed by shared address
#define WGEMU_DL_SMEM_BYTES 233472u  // 228 KB of shared-window addresses
#define WGEMU_DL_MAX_THREADS 1024u
#define WGEMU_DL_HDR_BYTES 32u
#define WGEMU_DL_MAGIC 0x31454757u  // "WGE1"
#define WGEMU_DL_POISON 0xFFFFFFFFu
#define WGEMU_DL_SLOT_TGROUP 64u
#define WGEMU_DL_SLOT_TCOUNT (WGEMU_DL_SLOT_TGROUP + 4u * WGEMU_DL_MAX_THREADS)
#define WGEMU_DL_SLOT_OWNER (WGEMU_DL_SLOT_TCOUNT + 4u * WGEMU_DL_MAX_THREADS)
#define WGEMU_DL_SLOT_RECORDS 125184u  // WGEMU_DL_SLOT_OWNER + SMEM_BYTES / 2, rounded to 256
#define WGEMU_DL_REC_BYTES 32u
#define WGEMU_DL_SLOT_BYTES(nrec) \
  (WGEMU_DL_SLOT_RECORDS + WGEMU_DL_REC_BYTES * (nrec) + WGEMU_DL_SMEM_BYTES)
#define WGEMU_DL_LIVE 0x1u
#define WGEMU_DL_BOUND 0x2u
#define WGEMU_DL_PARITY 0x4u
#define WGEMU_DL_CP 0x8u
#define WGEMU_DL_OWNER_SHIFT 4
#define WGEMU_AF_DL_ALL 0x1u  // wgemu_dl_cp_wait: cp.async.wait_all

// ---- E2 counters (keep in sync with tools/ptxemu/instrument.py and host/shim.c) ----------
// Module block __wgemu_global_count_v1 (WGEMU_CNT_BYTES, .weak .global, zeroed by the shim after
// each counted launch): u64 counters at the byte offsets below, then WGEMU_CNT_WGMMA_SLOTS
// entries {u32 code, u32 pad, u64 count} of an open-addressing table keyed by the wgmma code
// (bit 0 valid, 1 sparse, 2-10 N, 11-19 K, 20-23 D type, 24-27 A type, 28-31 B type; type
// numbers in instrument.WGMMA_TYPES).
#define WGEMU_CNT_BYTES 1088u
#define WGEMU_CNT_BULK 0u
#define WGEMU_CNT_TMA 8u
#define WGEMU_CNT_CPASYNC 16u
#define WGEMU_CNT_LDG 24u
#define WGEMU_CNT_MBAR_WAITS 32u
#define WGEMU_CNT_WGMMA_OVERFLOW 40u
#define WGEMU_CNT_WGMMA_TABLE 64u
#define WGEMU_CNT_WGMMA_SLOTS 64u

// mbarrier word layout.
#define WGEMU_MB_PENDING_MASK 0xFFFFFull
#define WGEMU_MB_EXPECTED_SHIFT 20
#define WGEMU_MB_TX_SHIFT 40
#define WGEMU_MB_TX_MASK 0x1FFFFFull
#define WGEMU_MB_PHASE_BIT 61
#define WGEMU_MB_VALID_BIT 62
#define WGEMU_MB_V1_BIT 63
#define WGEMU_MB_TX_LIMIT ((1 << 20) - 1)
#define WGEMU_MB_COUNT_MAX_V0 ((1u << 20) - 1u)
#define WGEMU_MB_COUNT_MAX_V1 ((1u << 9) - 1u)

#if defined(__CUDACC__)

namespace wgemu {
namespace async {

// ---- raw memory access ------------------------------------------------------------------------
// WGEMU_SMEM_GLOBAL (E4 shadow shared memory mode, include/wgmma_emu/e4_device.cuh): every
// shared address is a virtual offset into the CTA's slot in global memory; the accessors check
// the offset against the window, then access the global state space.
#ifdef WGEMU_SMEM_GLOBAL
WGEMU_DEV uint64_t shared_ld(uint32_t a, int n) { return e4::ld(a, n); }
WGEMU_DEV void shared_st(uint32_t a, int n, uint64_t v) { e4::st(a, n, v); }
WGEMU_DEV void shared_ld16(uint32_t a, uint32_t* w) { e4::ld16(a, w); }
WGEMU_DEV void shared_st16(uint32_t a, const uint32_t* w) { e4::st16(a, w); }
WGEMU_DEV void shared_st_u64x2(uint32_t a, uint64_t lo, uint64_t hi) { e4::st_u64x2(a, lo, hi); }
// Address operand of an atomic / reduction / mbarrier access of `n` bytes, in the state space
// WGEMU_SH_SP: a checked 64-bit global address ("l"); without E4 the 32-bit shared address ("r").
typedef uint64_t sh_addr_t;
WGEMU_DEV sh_addr_t sh_addr(uint32_t a, uint32_t n) { return e4::addr(a, n); }
#define WGEMU_SH_SP "global"
#define WGEMU_SH_C "l"
#else
WGEMU_DEV uint64_t shared_ld(uint32_t a, int n) {
  uint64_t v = 0;
  if (n == 1) {
    uint16_t t;
    asm volatile("ld.shared::cta.u8 %0, [%1];" : "=h"(t) : "r"(a) : "memory");
    v = t & 0xFFu;
  } else if (n == 2) {
    uint16_t t;
    asm volatile("ld.shared::cta.u16 %0, [%1];" : "=h"(t) : "r"(a) : "memory");
    v = t;
  } else if (n == 4) {
    uint32_t t;
    asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(t) : "r"(a) : "memory");
    v = t;
  } else {
    asm volatile("ld.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"(a) : "memory");
  }
  return v;
}

WGEMU_DEV void shared_st(uint32_t a, int n, uint64_t v) {
  if (n == 1) {
    asm volatile("st.shared::cta.u8 [%0], %1;" ::"r"(a), "h"((uint16_t)(v & 0xFFu)) : "memory");
  } else if (n == 2) {
    asm volatile("st.shared::cta.u16 [%0], %1;" ::"r"(a), "h"((uint16_t)v) : "memory");
  } else if (n == 4) {
    asm volatile("st.shared::cta.u32 [%0], %1;" ::"r"(a), "r"((uint32_t)v) : "memory");
  } else {
    asm volatile("st.shared::cta.u64 [%0], %1;" ::"r"(a), "l"(v) : "memory");
  }
}

WGEMU_DEV void shared_ld16(uint32_t a, uint32_t* w) {
  asm volatile("ld.shared::cta.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
               : "r"(a)
               : "memory");
}

WGEMU_DEV void shared_st16(uint32_t a, const uint32_t* w) {
  asm volatile("st.shared::cta.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(w[0]), "r"(w[1]),
               "r"(w[2]), "r"(w[3])
               : "memory");
}

WGEMU_DEV void shared_st_u64x2(uint32_t a, uint64_t lo, uint64_t hi) {
  asm volatile("st.shared::cta.v2.u64 [%0], {%1, %2};" ::"r"(a), "l"(lo), "l"(hi) : "memory");
}

typedef uint32_t sh_addr_t;
WGEMU_DEV sh_addr_t sh_addr(uint32_t a, uint32_t n) {
  (void)n;
  return a;
}
#define WGEMU_SH_SP "shared::cta"
#define WGEMU_SH_C "r"
#endif

// Generic addressing (global memory, and tensor maps in .param/.const/.global/.shared).
WGEMU_DEV uint64_t gen_ld(uint64_t a, int n) {
  uint64_t v = 0;
  if (n == 1) {
    uint16_t t;
    asm volatile("ld.u8 %0, [%1];" : "=h"(t) : "l"(a) : "memory");
    v = t & 0xFFu;
  } else if (n == 2) {
    uint16_t t;
    asm volatile("ld.u16 %0, [%1];" : "=h"(t) : "l"(a) : "memory");
    v = t;
  } else if (n == 4) {
    uint32_t t;
    asm volatile("ld.u32 %0, [%1];" : "=r"(t) : "l"(a) : "memory");
    v = t;
  } else {
    asm volatile("ld.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  }
  return v;
}

WGEMU_DEV void gen_st(uint64_t a, int n, uint64_t v) {
  if (n == 1) {
    asm volatile("st.u8 [%0], %1;" ::"l"(a), "h"((uint16_t)(v & 0xFFu)) : "memory");
  } else if (n == 2) {
    asm volatile("st.u16 [%0], %1;" ::"l"(a), "h"((uint16_t)v) : "memory");
  } else if (n == 4) {
    asm volatile("st.u32 [%0], %1;" ::"l"(a), "r"((uint32_t)v) : "memory");
  } else {
    asm volatile("st.u64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
  }
}

WGEMU_DEV void gen_ld16(uint64_t a, uint32_t* w) {
  asm volatile("ld.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
               : "l"(a)
               : "memory");
}

WGEMU_DEV void gen_st16(uint64_t a, const uint32_t* w) {
  asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(a), "r"(w[0]), "r"(w[1]), "r"(w[2]),
               "r"(w[3])
               : "memory");
}

WGEMU_DEV uint64_t to_global(uint64_t a) {
  uint64_t g;
  asm("cvta.to.global.u64 %0, %1;" : "=l"(g) : "l"(a));
  return g;
}

WGEMU_DEV void trap() { asm volatile("trap;"); }

WGEMU_DEV uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

WGEMU_DEV void nanosleep_ns(uint32_t ns) { asm volatile("nanosleep.u32 %0;" ::"r"(ns)); }

// ---- mbarrier word ------------------------------------------------------------------------------
WGEMU_DEV uint64_t mb_load_acquire(uint32_t bar) {
  uint64_t v;
#ifdef WGEMU_SMEM_GLOBAL
  asm volatile("ld.acquire.cta.global.u64 %0, [%1];" : "=l"(v) : "l"(sh_addr(bar, 8u)) : "memory");
#else
  asm volatile("ld.acquire.cta.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"(bar) : "memory");
#endif
  return v;
}

WGEMU_DEV uint64_t mb_load_relaxed(uint32_t bar) {
  uint64_t v;
#ifdef WGEMU_SMEM_GLOBAL
  asm volatile("ld.relaxed.cta.global.u64 %0, [%1];" : "=l"(v) : "l"(sh_addr(bar, 8u)) : "memory");
#else
  asm volatile("ld.relaxed.cta.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"(bar) : "memory");
#endif
  return v;
}

WGEMU_DEV uint64_t mb_cas(uint32_t bar, uint64_t cmp, uint64_t val) {
  uint64_t old;
#ifdef WGEMU_SMEM_GLOBAL
  asm volatile("atom.acq_rel.cta.global.cas.b64 %0, [%1], %2, %3;"
               : "=l"(old)
               : "l"(sh_addr(bar, 8u)), "l"(cmp), "l"(val)
               : "memory");
#else
  asm volatile("atom.acq_rel.cta.shared::cta.cas.b64 %0, [%1], %2, %3;"
               : "=l"(old)
               : "r"(bar), "l"(cmp), "l"(val)
               : "memory");
#endif
  return old;
}

WGEMU_DEV uint32_t mb_pending(uint64_t w) { return (uint32_t)(w & WGEMU_MB_PENDING_MASK); }
WGEMU_DEV uint32_t mb_expected(uint64_t w) {
  return (uint32_t)((w >> WGEMU_MB_EXPECTED_SHIFT) & WGEMU_MB_PENDING_MASK);
}
WGEMU_DEV int32_t mb_tx(uint64_t w) {
  int32_t t = (int32_t)((w >> WGEMU_MB_TX_SHIFT) & WGEMU_MB_TX_MASK);
  return (t ^ 0x100000) - 0x100000;  // sign-extend 21 bits
}
WGEMU_DEV uint32_t mb_phase(uint64_t w) { return (uint32_t)((w >> WGEMU_MB_PHASE_BIT) & 1u); }
WGEMU_DEV bool mb_valid(uint64_t w) { return ((w >> WGEMU_MB_VALID_BIT) & 1u) != 0; }

WGEMU_DEV uint64_t mb_pack(uint32_t pending, uint32_t expected, int32_t tx, uint32_t phase,
                           uint64_t keep_flags) {
  return (uint64_t)(pending & WGEMU_MB_PENDING_MASK) |
         ((uint64_t)(expected & WGEMU_MB_PENDING_MASK) << WGEMU_MB_EXPECTED_SHIFT) |
         (((uint64_t)(uint32_t)tx & WGEMU_MB_TX_MASK) << WGEMU_MB_TX_SHIFT) |
         ((uint64_t)(phase & 1u) << WGEMU_MB_PHASE_BIT) | keep_flags;
}

// ---- tensor-map element conversion ------------------------------------------------------------
WGEMU_DEV uint32_t f32_flush(uint32_t x) {
  return ((x & 0x7F800000u) == 0u) ? (x & 0x80000000u) : x;
}

// Round f32 bits to tf32 (10 explicit mantissa bits), nearest even; NaN stays NaN.
WGEMU_DEV uint32_t f32_to_tf32_rne(uint32_t x) {
  if ((x & 0x7F800000u) == 0x7F800000u) {
    if ((x & 0x007FFFFFu) == 0u) return x;  // inf
    uint32_t r = x & 0xFFFFE000u;
    return ((r & 0x007FFFFFu) == 0u) ? (r | 0x00400000u) : r;
  }
  uint32_t r = x + 0xFFFu + ((x >> 13) & 1u);
  return r & 0xFFFFE000u;
}

// Element value transform applied by every TMA transfer for the tensor-map element type [inferred:
// ptx.txt 3156-3157 lists ".tf32 ... (rounded to nearest even)"; FTZ types flush subnormals].
WGEMU_DEV uint64_t tm_convert(uint64_t v, int etype) {
  switch (etype) {
    case WGEMU_TM_F32_FTZ:
      return f32_flush((uint32_t)v);
    case WGEMU_TM_TF32:
      return f32_to_tf32_rne((uint32_t)v);
    case WGEMU_TM_TF32_FTZ:
      return f32_to_tf32_rne(f32_flush((uint32_t)v));
    default:
      return v;
  }
}

WGEMU_DEV bool tm_needs_convert(int etype) {
  return etype == WGEMU_TM_F32_FTZ || etype == WGEMU_TM_TF32 || etype == WGEMU_TM_TF32_FTZ;
}

// OOB-NaN fill pattern [inferred: canonical NaN of the element width].
WGEMU_DEV uint64_t tm_nan_fill(int etype) {
  switch (etype) {
    case WGEMU_TM_F16:
    case WGEMU_TM_BF16:
      return 0x7FFFull;
    case WGEMU_TM_F32:
    case WGEMU_TM_F32_FTZ:
    case WGEMU_TM_TF32:
    case WGEMU_TM_TF32_FTZ:
      return 0x7FFFFFFFull;
    case WGEMU_TM_F64:
      return 0x7FFFFFFFFFFFFFFFull;
    default:
      return 0;
  }
}

}  // namespace async
}  // namespace wgemu

#endif  // __CUDACC__
