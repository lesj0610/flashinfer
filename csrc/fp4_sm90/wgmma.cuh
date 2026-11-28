/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// PTX wrappers of the SM90 FP4 GEMM: mbarrier, async copies, shared-memory access, fences and
// wgmma (A from registers, B from shared memory, FP32 accumulators).
//
// Only instructions of the wgmma emulator inventory are used (tools/ptxemu/inventory.tsv), so the
// same source also builds with nvcc_emu (sm_90a PTX -> sm_80). setmaxnreg is a no-op there.
#pragma once

#include <cuda.h>

#include <cstdint>

namespace flashinfer {
namespace fp4_sm90 {
namespace ptx {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------------------------
// mbarrier
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

// Arrive when pred != 0, without a branch: control flow between wgmma.mma_async and the
// wgmma.wait_group that retires it makes ptxas serialize every wgmma (C7513).
__device__ __forceinline__ void mbar_arrive_if(uint32_t bar, uint32_t pred) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %1, 0;\n"
      "@p mbarrier.arrive.shared::cta.b64 _, [%0];\n"
      "}\n" ::"r"(bar),
      "r"(pred)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "FP4SM90_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra FP4SM90_WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

// Generic-proxy shared-memory writes (cp.async, st.shared) -> async-proxy reads (wgmma B).
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <int R>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R));
}

template <int R>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R));
}

// ---------------------------------------------------------------------------------------------
// Async copies. cp.async zero-fills the bytes past src_size (src_size 0: nothing is read, the
// address only has to be a valid global address).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src, uint32_t src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(src_size)
               : "memory");
}

__device__ __forceinline__ void cp_async_8(uint32_t dst, const void* src, uint32_t src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(src_size)
               : "memory");
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint32_t src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_size)
               : "memory");
}

// Arrive on `bar` once every cp.async of this thread issued so far has completed. The barrier's
// expected count must include this arrival (.noinc).
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_5d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                       int c3, int c4, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}

// ---------------------------------------------------------------------------------------------
// Shared-memory access and byte permutes
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t lds_u16(uint32_t addr) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t lds_u8(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ void sts16(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(uint16_t(v)) : "memory");
}

__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ void sts64(uint32_t addr, uint32_t v0, uint32_t v1) {
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::"r"(addr), "r"(v0), "r"(v1) : "memory");
}

// prmt.b32 default mode: selector nibble bits 2:0 pick a byte of {b, a}, bit 3 replicates its sign.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// ---------------------------------------------------------------------------------------------
// wgmma
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(Pending) : "memory");
}

// Keeps the compiler from moving accumulator accesses across wgmma.wait_group / wgmma.fence.
template <int NR>
__device__ __forceinline__ void fence_operands(float (&d)[NR]) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

// Shared-memory descriptor of the B operand: K-major, 128-byte swizzle, rows of 128 bytes
// (64 BF16/FP16 values), 8-row groups 1024 bytes apart (SBO), LBO unused (1).
// One B stage = two 64-value halves of N rows (params.cuh b_tile_byte); k16 step s of the K-128
// stage starts at half s / 4, byte 32 * (s % 4) of the swizzle atom.
constexpr uint64_t kBDescHi = (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
constexpr uint64_t kBDescLbo = uint64_t(1) << 16;

template <int N>
__device__ __forceinline__ uint64_t b_desc(uint32_t b_stage, int s) {
  const uint32_t addr = b_stage + uint32_t(s >> 2) * uint32_t(N * 128) + uint32_t(s & 3) * 32u;
  return kBDescHi | kBDescLbo | uint64_t((addr & 0x3FFFFu) >> 4);
}

// wgmma.mma_async m64nNk16, f32 += {bf16|f16} (A registers) x {bf16|f16} (B descriptor), scale-d 1.
// The accumulators are d[OFF .. OFF + N / 2) (OFF selects the m64 block of a CTA_M 256 tile).
// Generated text (one specialization per N and operand type).
template <int N, bool F16>
struct WgmmaRS;

template <>
struct WgmmaRS<8, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 4 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, %8, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<8, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 4 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, %8, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<16, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 8 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<16, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 8 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<32, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 16 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<32, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 16 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<64, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 32 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<64, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 32 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<128, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 64 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31]), "+f"(d[OFF + 32]), "+f"(d[OFF + 33]),
          "+f"(d[OFF + 34]), "+f"(d[OFF + 35]), "+f"(d[OFF + 36]), "+f"(d[OFF + 37]),
          "+f"(d[OFF + 38]), "+f"(d[OFF + 39]), "+f"(d[OFF + 40]), "+f"(d[OFF + 41]),
          "+f"(d[OFF + 42]), "+f"(d[OFF + 43]), "+f"(d[OFF + 44]), "+f"(d[OFF + 45]),
          "+f"(d[OFF + 46]), "+f"(d[OFF + 47]), "+f"(d[OFF + 48]), "+f"(d[OFF + 49]),
          "+f"(d[OFF + 50]), "+f"(d[OFF + 51]), "+f"(d[OFF + 52]), "+f"(d[OFF + 53]),
          "+f"(d[OFF + 54]), "+f"(d[OFF + 55]), "+f"(d[OFF + 56]), "+f"(d[OFF + 57]),
          "+f"(d[OFF + 58]), "+f"(d[OFF + 59]), "+f"(d[OFF + 60]), "+f"(d[OFF + 61]),
          "+f"(d[OFF + 62]), "+f"(d[OFF + 63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<128, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 64 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31]), "+f"(d[OFF + 32]), "+f"(d[OFF + 33]),
          "+f"(d[OFF + 34]), "+f"(d[OFF + 35]), "+f"(d[OFF + 36]), "+f"(d[OFF + 37]),
          "+f"(d[OFF + 38]), "+f"(d[OFF + 39]), "+f"(d[OFF + 40]), "+f"(d[OFF + 41]),
          "+f"(d[OFF + 42]), "+f"(d[OFF + 43]), "+f"(d[OFF + 44]), "+f"(d[OFF + 45]),
          "+f"(d[OFF + 46]), "+f"(d[OFF + 47]), "+f"(d[OFF + 48]), "+f"(d[OFF + 49]),
          "+f"(d[OFF + 50]), "+f"(d[OFF + 51]), "+f"(d[OFF + 52]), "+f"(d[OFF + 53]),
          "+f"(d[OFF + 54]), "+f"(d[OFF + 55]), "+f"(d[OFF + 56]), "+f"(d[OFF + 57]),
          "+f"(d[OFF + 58]), "+f"(d[OFF + 59]), "+f"(d[OFF + 60]), "+f"(d[OFF + 61]),
          "+f"(d[OFF + 62]), "+f"(d[OFF + 63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<256, false> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 128 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, "
        "%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
        "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, "
        "%107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
        "%122, %123, %124, %125, %126, %127}, "
        "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31]), "+f"(d[OFF + 32]), "+f"(d[OFF + 33]),
          "+f"(d[OFF + 34]), "+f"(d[OFF + 35]), "+f"(d[OFF + 36]), "+f"(d[OFF + 37]),
          "+f"(d[OFF + 38]), "+f"(d[OFF + 39]), "+f"(d[OFF + 40]), "+f"(d[OFF + 41]),
          "+f"(d[OFF + 42]), "+f"(d[OFF + 43]), "+f"(d[OFF + 44]), "+f"(d[OFF + 45]),
          "+f"(d[OFF + 46]), "+f"(d[OFF + 47]), "+f"(d[OFF + 48]), "+f"(d[OFF + 49]),
          "+f"(d[OFF + 50]), "+f"(d[OFF + 51]), "+f"(d[OFF + 52]), "+f"(d[OFF + 53]),
          "+f"(d[OFF + 54]), "+f"(d[OFF + 55]), "+f"(d[OFF + 56]), "+f"(d[OFF + 57]),
          "+f"(d[OFF + 58]), "+f"(d[OFF + 59]), "+f"(d[OFF + 60]), "+f"(d[OFF + 61]),
          "+f"(d[OFF + 62]), "+f"(d[OFF + 63]), "+f"(d[OFF + 64]), "+f"(d[OFF + 65]),
          "+f"(d[OFF + 66]), "+f"(d[OFF + 67]), "+f"(d[OFF + 68]), "+f"(d[OFF + 69]),
          "+f"(d[OFF + 70]), "+f"(d[OFF + 71]), "+f"(d[OFF + 72]), "+f"(d[OFF + 73]),
          "+f"(d[OFF + 74]), "+f"(d[OFF + 75]), "+f"(d[OFF + 76]), "+f"(d[OFF + 77]),
          "+f"(d[OFF + 78]), "+f"(d[OFF + 79]), "+f"(d[OFF + 80]), "+f"(d[OFF + 81]),
          "+f"(d[OFF + 82]), "+f"(d[OFF + 83]), "+f"(d[OFF + 84]), "+f"(d[OFF + 85]),
          "+f"(d[OFF + 86]), "+f"(d[OFF + 87]), "+f"(d[OFF + 88]), "+f"(d[OFF + 89]),
          "+f"(d[OFF + 90]), "+f"(d[OFF + 91]), "+f"(d[OFF + 92]), "+f"(d[OFF + 93]),
          "+f"(d[OFF + 94]), "+f"(d[OFF + 95]), "+f"(d[OFF + 96]), "+f"(d[OFF + 97]),
          "+f"(d[OFF + 98]), "+f"(d[OFF + 99]), "+f"(d[OFF + 100]), "+f"(d[OFF + 101]),
          "+f"(d[OFF + 102]), "+f"(d[OFF + 103]), "+f"(d[OFF + 104]), "+f"(d[OFF + 105]),
          "+f"(d[OFF + 106]), "+f"(d[OFF + 107]), "+f"(d[OFF + 108]), "+f"(d[OFF + 109]),
          "+f"(d[OFF + 110]), "+f"(d[OFF + 111]), "+f"(d[OFF + 112]), "+f"(d[OFF + 113]),
          "+f"(d[OFF + 114]), "+f"(d[OFF + 115]), "+f"(d[OFF + 116]), "+f"(d[OFF + 117]),
          "+f"(d[OFF + 118]), "+f"(d[OFF + 119]), "+f"(d[OFF + 120]), "+f"(d[OFF + 121]),
          "+f"(d[OFF + 122]), "+f"(d[OFF + 123]), "+f"(d[OFF + 124]), "+f"(d[OFF + 125]),
          "+f"(d[OFF + 126]), "+f"(d[OFF + 127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};

template <>
struct WgmmaRS<256, true> {
  template <int OFF, int TOT>
  static __device__ __forceinline__ void run(float (&d)[TOT], const uint32_t (&a)[4],
                                             uint64_t desc) {
    static_assert(OFF + 128 <= TOT, "accumulator range");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
        "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
        "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
        "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, "
        "%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, "
        "%91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, "
        "%107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
        "%122, %123, %124, %125, %126, %127}, "
        "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[OFF + 0]), "+f"(d[OFF + 1]), "+f"(d[OFF + 2]), "+f"(d[OFF + 3]), "+f"(d[OFF + 4]),
          "+f"(d[OFF + 5]), "+f"(d[OFF + 6]), "+f"(d[OFF + 7]), "+f"(d[OFF + 8]), "+f"(d[OFF + 9]),
          "+f"(d[OFF + 10]), "+f"(d[OFF + 11]), "+f"(d[OFF + 12]), "+f"(d[OFF + 13]),
          "+f"(d[OFF + 14]), "+f"(d[OFF + 15]), "+f"(d[OFF + 16]), "+f"(d[OFF + 17]),
          "+f"(d[OFF + 18]), "+f"(d[OFF + 19]), "+f"(d[OFF + 20]), "+f"(d[OFF + 21]),
          "+f"(d[OFF + 22]), "+f"(d[OFF + 23]), "+f"(d[OFF + 24]), "+f"(d[OFF + 25]),
          "+f"(d[OFF + 26]), "+f"(d[OFF + 27]), "+f"(d[OFF + 28]), "+f"(d[OFF + 29]),
          "+f"(d[OFF + 30]), "+f"(d[OFF + 31]), "+f"(d[OFF + 32]), "+f"(d[OFF + 33]),
          "+f"(d[OFF + 34]), "+f"(d[OFF + 35]), "+f"(d[OFF + 36]), "+f"(d[OFF + 37]),
          "+f"(d[OFF + 38]), "+f"(d[OFF + 39]), "+f"(d[OFF + 40]), "+f"(d[OFF + 41]),
          "+f"(d[OFF + 42]), "+f"(d[OFF + 43]), "+f"(d[OFF + 44]), "+f"(d[OFF + 45]),
          "+f"(d[OFF + 46]), "+f"(d[OFF + 47]), "+f"(d[OFF + 48]), "+f"(d[OFF + 49]),
          "+f"(d[OFF + 50]), "+f"(d[OFF + 51]), "+f"(d[OFF + 52]), "+f"(d[OFF + 53]),
          "+f"(d[OFF + 54]), "+f"(d[OFF + 55]), "+f"(d[OFF + 56]), "+f"(d[OFF + 57]),
          "+f"(d[OFF + 58]), "+f"(d[OFF + 59]), "+f"(d[OFF + 60]), "+f"(d[OFF + 61]),
          "+f"(d[OFF + 62]), "+f"(d[OFF + 63]), "+f"(d[OFF + 64]), "+f"(d[OFF + 65]),
          "+f"(d[OFF + 66]), "+f"(d[OFF + 67]), "+f"(d[OFF + 68]), "+f"(d[OFF + 69]),
          "+f"(d[OFF + 70]), "+f"(d[OFF + 71]), "+f"(d[OFF + 72]), "+f"(d[OFF + 73]),
          "+f"(d[OFF + 74]), "+f"(d[OFF + 75]), "+f"(d[OFF + 76]), "+f"(d[OFF + 77]),
          "+f"(d[OFF + 78]), "+f"(d[OFF + 79]), "+f"(d[OFF + 80]), "+f"(d[OFF + 81]),
          "+f"(d[OFF + 82]), "+f"(d[OFF + 83]), "+f"(d[OFF + 84]), "+f"(d[OFF + 85]),
          "+f"(d[OFF + 86]), "+f"(d[OFF + 87]), "+f"(d[OFF + 88]), "+f"(d[OFF + 89]),
          "+f"(d[OFF + 90]), "+f"(d[OFF + 91]), "+f"(d[OFF + 92]), "+f"(d[OFF + 93]),
          "+f"(d[OFF + 94]), "+f"(d[OFF + 95]), "+f"(d[OFF + 96]), "+f"(d[OFF + 97]),
          "+f"(d[OFF + 98]), "+f"(d[OFF + 99]), "+f"(d[OFF + 100]), "+f"(d[OFF + 101]),
          "+f"(d[OFF + 102]), "+f"(d[OFF + 103]), "+f"(d[OFF + 104]), "+f"(d[OFF + 105]),
          "+f"(d[OFF + 106]), "+f"(d[OFF + 107]), "+f"(d[OFF + 108]), "+f"(d[OFF + 109]),
          "+f"(d[OFF + 110]), "+f"(d[OFF + 111]), "+f"(d[OFF + 112]), "+f"(d[OFF + 113]),
          "+f"(d[OFF + 114]), "+f"(d[OFF + 115]), "+f"(d[OFF + 116]), "+f"(d[OFF + 117]),
          "+f"(d[OFF + 118]), "+f"(d[OFF + 119]), "+f"(d[OFF + 120]), "+f"(d[OFF + 121]),
          "+f"(d[OFF + 122]), "+f"(d[OFF + 123]), "+f"(d[OFF + 124]), "+f"(d[OFF + 125]),
          "+f"(d[OFF + 126]), "+f"(d[OFF + 127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(1));
  }
};
}  // namespace ptx
}  // namespace fp4_sm90
}  // namespace flashinfer
