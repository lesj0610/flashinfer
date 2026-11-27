// E4 shadow shared memory mode (fork-only). Only compiled with -DWGEMU_SMEM_GLOBAL; without the
// define this header is empty and the library PTX is unchanged.
//
// The whole CTA shared window lives in a per-CTA slot of global memory ("shadow"). Every 32-bit
// shared address of the translated program is a virtual offset inside the slot. The only real
// shared memory of an E4 kernel is the 32-byte header __wgemu_e4_hdr_v1 (defined below, global
// namespace, unmangled). Contract: the E4 ABI document (E4_abi.md) sections 2, 3 and 6.
//
// Header layout (raw ld/st/atom.shared on __wgemu_e4_hdr_v1 only):
//   +0 u64 slot base (generic address of the leased slot; 0 = no slot)   +8 u32 window end
//   +12 u32 dynamic bytes   +16 u32 alive thread count   +20 u32 slot index
//   +24 u32 magic WGEMU_E4_MAGIC (written last by thread 0 before bar.sync)   +28 u32 reserved
// Slot bytes [0, WGEMU_E4_ORIGIN) are never reachable by a shared access (the accessors reject
// offsets below the origin); the lib keeps the lease and stats addresses of the CTA there:
//   slot+0 u64 generic address of the lease word of this slot, slot+8 u64 generic stats address.
#pragma once

#ifdef WGEMU_SMEM_GLOBAL

#include <stdint.h>
#include <stdio.h>

#include "types.h"

#define WGEMU_E4_ORIGIN 0x400u
#define WGEMU_E4_WINDOW_MAX 0x40000u
#define WGEMU_E4_STRIDE 0x40000u
#define WGEMU_E4_SM90_BLOCK_MAX 232448u
#define WGEMU_E4_MAGIC 0x57474534u  // "WGE4"
// Header field offsets.
#define WGEMU_E4_H_BASE 0
#define WGEMU_E4_H_WINDOW 8
#define WGEMU_E4_H_DYN 12
#define WGEMU_E4_H_ALIVE 16
#define WGEMU_E4_H_SLOT 20
#define WGEMU_E4_H_MAGIC 24
// Slot-private bytes (below the origin).
#define WGEMU_E4_S_LEASE 0
#define WGEMU_E4_S_STATS 8

#if defined(__CUDACC__)

// The one real shared object of an E4 build. extern "C"-style unmangled name (namespace scope
// variable: nvcc emits `.visible .shared .align 16 .b8 __wgemu_e4_hdr_v1[32]`).
__shared__ __align__(16) unsigned long long __wgemu_e4_hdr_v1[4];

namespace wgemu {
namespace e4 {

// ---- raw header access (the only shared-space instructions of the E4 library) -----------------
template <int OFF>
WGEMU_DEV uint32_t hdr_ld32() {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [__wgemu_e4_hdr_v1+%1];" : "=r"(v) : "n"(OFF) : "memory");
  return v;
}

template <int OFF>
WGEMU_DEV uint64_t hdr_ld64() {
  uint64_t v;
  asm volatile("ld.shared.u64 %0, [__wgemu_e4_hdr_v1+%1];" : "=l"(v) : "n"(OFF) : "memory");
  return v;
}

template <int OFF>
WGEMU_DEV void hdr_st32(uint32_t v) {
  asm volatile("st.shared.u32 [__wgemu_e4_hdr_v1+%0], %1;" ::"n"(OFF), "r"(v) : "memory");
}

template <int OFF>
WGEMU_DEV void hdr_st64(uint64_t v) {
  asm volatile("st.shared.u64 [__wgemu_e4_hdr_v1+%0], %1;" ::"n"(OFF), "l"(v) : "memory");
}

// Exit rendezvous: acq_rel at cta scope on the header alive counter. Returns the old value.
WGEMU_DEV uint32_t hdr_alive_dec() {
  uint32_t old;
  asm volatile("atom.acq_rel.cta.shared.add.u32 %0, [__wgemu_e4_hdr_v1+%1], %2;"
               : "=r"(old)
               : "n"(WGEMU_E4_H_ALIVE), "r"(0xFFFFFFFFu)
               : "memory");
  return old;
}

// ---- bounds-checked mapping of a virtual offset -------------------------------------------------
__device__ __noinline__ void oob(uint32_t off, uint32_t width, uint32_t window) {
  printf(
      "WGEMU-ERROR e4 shared-access-out-of-window off=0x%08x width=%u window=0x%08x origin=0x%x "
      "tid=%u block=(%u,%u,%u)\n",
      off, width, window, WGEMU_E4_ORIGIN,
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z), blockIdx.x, blockIdx.y,
      blockIdx.z);
  __trap();
}

// Generic (global) address of the `n` bytes at virtual offset `off`; traps outside
// [WGEMU_E4_ORIGIN, window). Pure reads of the header: the header does not change between the
// barrier of wgemu_e4_enter and wgemu_e4_exit, so the compiler may share them.
WGEMU_DEV uint64_t addr(uint32_t off, uint32_t n) {
  uint64_t base;
  uint32_t window;
  asm("ld.shared.u64 %0, [__wgemu_e4_hdr_v1];" : "=l"(base));
  asm("ld.shared.u32 %0, [__wgemu_e4_hdr_v1+8];" : "=r"(window));
  if (off < WGEMU_E4_ORIGIN || (uint64_t)off + n > (uint64_t)window) oob(off, n, window);
  return base + off;
}

// ---- global-state-space data accessors (same widths and forms as the shared ones) --------------
WGEMU_DEV uint64_t gld(uint64_t g, int n) {
  uint64_t v = 0;
  if (n == 1) {
    uint16_t t;
    asm volatile("ld.global.u8 %0, [%1];" : "=h"(t) : "l"(g) : "memory");
    v = t & 0xFFu;
  } else if (n == 2) {
    uint16_t t;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(t) : "l"(g) : "memory");
    v = t;
  } else if (n == 4) {
    uint32_t t;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(t) : "l"(g) : "memory");
    v = t;
  } else {
    asm volatile("ld.global.u64 %0, [%1];" : "=l"(v) : "l"(g) : "memory");
  }
  return v;
}

WGEMU_DEV void gst(uint64_t g, int n, uint64_t v) {
  if (n == 1) {
    asm volatile("st.global.u8 [%0], %1;" ::"l"(g), "h"((uint16_t)(v & 0xFFu)) : "memory");
  } else if (n == 2) {
    asm volatile("st.global.u16 [%0], %1;" ::"l"(g), "h"((uint16_t)v) : "memory");
  } else if (n == 4) {
    asm volatile("st.global.u32 [%0], %1;" ::"l"(g), "r"((uint32_t)v) : "memory");
  } else {
    asm volatile("st.global.u64 [%0], %1;" ::"l"(g), "l"(v) : "memory");
  }
}

WGEMU_DEV uint32_t width_of(int n) { return (n == 1 || n == 2 || n == 4) ? (uint32_t)n : 8u; }

// Shared-memory replacements: virtual offset in, bounds check, global access.
WGEMU_DEV uint64_t ld(uint32_t a, int n) { return gld(addr(a, width_of(n)), n); }
WGEMU_DEV void st(uint32_t a, int n, uint64_t v) { gst(addr(a, width_of(n)), n, v); }

WGEMU_DEV void ld16(uint32_t a, uint32_t* w) {
  const uint64_t g = addr(a, 16u);
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
               : "l"(g)
               : "memory");
}

WGEMU_DEV void st16(uint32_t a, const uint32_t* w) {
  const uint64_t g = addr(a, 16u);
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(g), "r"(w[0]), "r"(w[1]), "r"(w[2]),
               "r"(w[3])
               : "memory");
}

WGEMU_DEV void st_u64x2(uint32_t a, uint64_t lo, uint64_t hi) {
  const uint64_t g = addr(a, 16u);
  asm volatile("st.global.v2.u64 [%0], {%1, %2};" ::"l"(g), "l"(lo), "l"(hi) : "memory");
}

// wgmma operand read (8, 16 or 32 bit element, zero extended).
WGEMU_DEV uint32_t lds(uint32_t a, int bits) {
  uint32_t v;
  const uint64_t g = addr(a, bits == 8 ? 1u : (bits == 16 ? 2u : 4u));
  if (bits == 8) {
    asm volatile("ld.global.u8 %0, [%1];" : "=r"(v) : "l"(g) : "memory");
  } else if (bits == 16) {
    asm volatile("ld.global.u16 %0, [%1];" : "=r"(v) : "l"(g) : "memory");
  } else {
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(g) : "memory");
  }
  return v;
}

}  // namespace e4
}  // namespace wgemu

#endif  // __CUDACC__
#endif  // WGEMU_SMEM_GLOBAL
