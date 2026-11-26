// End-to-end check of the sm_90 atom / red / multimem forms rewritten by
// tools/ptxemu/rules_misc.py (fork-only). Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Covered: atom/red .add.noftz.{bf16,bf16x2,f32} (global, generic, shared), vector
// atom/red (.v2/.v4 f32, .v8 bf16, .v2 bf16x2 max, .v4 f16 min, .v2 f16x2 add), .cluster scope,
// .shared::cluster space, multimem.ld_reduce / st / red.
//
// Every expectation is independent of the order in which threads arrive: additions use values
// whose partial sums are all exact (small integers in bf16 / f16 / f32, multiples of a subnormal
// in f32 so .noftz is visible), min / max use their order-free result, and the values returned
// by atom must be a permutation of the partial sums (each one exactly once), which also proves
// that every update was atomic.
//
// [inference] multimem with a single replica: the address is an ordinary global address; the
// emulator treats it as the only replica (PTX leaves non-multimem addresses undefined).
#include <cuda_fp16.h>
#include <math.h>

#include "misc_e2e.h"

using namespace misc_e2e;

typedef unsigned short u16;
typedef unsigned long long u64;

constexpr int kBlocks = 64, kThreads = 256, kNT = kBlocks * kThreads;
constexpr int kNC = 64;                // global counters; each receives kNT / kNC updates
constexpr int kPer = kNT / kNC;        // 256
constexpr int kSC = 8;                 // shared counters per block
constexpr int kSPer = kThreads / kSC;  // 32
constexpr uint32_t kSub = 0x200u;      // f32 subnormal 2^-140

#define TID() (blockIdx.x * blockDim.x + threadIdx.x)

// ---- bf16 / f32 .noftz add ------------------------------------------------------------------
__global__ void k_atom_bf16(u16* c, u16* old) {
  const int t = TID();
  u16 r;
  asm volatile("atom.global.add.noftz.bf16 %0, [%1], %2;"
               : "=h"(r)
               : "l"(c + t % kNC), "h"((u16)0x3F80));
  old[t] = r;
}
__global__ void k_atom_bf16x2(uint32_t* c, uint32_t* old) {
  const int t = TID();
  uint32_t r;
  asm volatile("atom.global.add.noftz.bf16x2 %0, [%1], %2;"
               : "=r"(r)
               : "l"(c + t % kNC), "r"(0x40003F80u));  // (lo 1.0, hi 2.0)
  old[t] = r;
}
__global__ void k_red_bf16(u16* c) {
  const int t = TID();
  asm volatile("red.global.add.noftz.bf16 [%0], %1;" ::"l"(c + t % kNC), "h"((u16)0x3F80));
}
__global__ void k_red_bf16x2(uint32_t* c) {
  const int t = TID();
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(c + t % kNC), "r"(0x40003F80u));
}
__global__ void k_atom_f32_global(uint32_t* c, uint32_t* old) {
  const int t = TID();
  uint32_t r;
  asm volatile("atom.global.add.noftz.f32 %0, [%1], %2;" : "=r"(r) : "l"(c + t % kNC), "r"(kSub));
  old[t] = r;
}
__global__ void k_atom_f32_generic(uint32_t* c, uint32_t* old) {
  const int t = TID();
  uint32_t r;
  asm volatile("atom.add.noftz.f32 %0, [%1], %2;" : "=r"(r) : "l"(c + t % kNC), "r"(kSub));
  old[t] = r;
}
__global__ void k_red_f32(uint32_t* c) {
  const int t = TID();
  asm volatile("red.global.add.noftz.f32 [%0], %1;" ::"l"(c + t % kNC), "r"(kSub));
}
__global__ void k_atom_f32_shared(uint32_t* fin, uint32_t* old) {
  __shared__ uint32_t s[kSC];
  if (threadIdx.x < kSC) s[threadIdx.x] = 0;
  __syncthreads();
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(&s[threadIdx.x % kSC]);
  uint32_t r;
  asm volatile("atom.shared.add.noftz.f32 %0, [%1], %2;" : "=r"(r) : "r"(a), "r"(kSub));
  old[TID()] = r;
  __syncthreads();
  if (threadIdx.x < kSC) fin[blockIdx.x * kSC + threadIdx.x] = s[threadIdx.x];
}

// ---- vector atom / red -------------------------------------------------------------------------
__global__ void k_atom_v2_f32(uint32_t* c, uint32_t* old) {
  const int t = TID();
  uint32_t r0, r1;
  asm volatile("atom.global.v2.f32.add {%0, %1}, [%2], {%3, %4};"
               : "=r"(r0), "=r"(r1)
               : "l"(c + 2 * (t % kNC)), "r"(0x3F800000u), "r"(0x40000000u));
  old[2 * t] = r0;
  old[2 * t + 1] = r1;
}
__global__ void k_red_v4_f32(uint32_t* c) {
  const int t = TID();
  asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(c + 4 * (t % kNC)),
               "r"(kSub), "r"(2 * kSub), "r"(3 * kSub), "r"(4 * kSub));
}
__global__ void k_atom_v8_bf16(u16* c, u16* old) {
  const int t = TID();
  const u16 one = 0x3F80;
  u16 r[8];
  asm volatile(
      "atom.global.v8.bf16.add.noftz {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], "
      "{%9, %10, %11, %12, %13, %14, %15, %16};"
      : "=h"(r[0]), "=h"(r[1]), "=h"(r[2]), "=h"(r[3]), "=h"(r[4]), "=h"(r[5]), "=h"(r[6]),
        "=h"(r[7])
      : "l"(c + 8 * (t % kNC)), "h"(one), "h"(one), "h"(one), "h"(one), "h"(one), "h"(one),
        "h"(one), "h"(one));
  for (int j = 0; j < 8; ++j) old[8 * t + j] = r[j];
}
// i = t / kNC in [0, 255]: word0 = (i, -(i + 1)), word1 = (i / 2, -(2 i + 2)), all bf16-exact.
__device__ uint32_t bf16_of(float x) { return __float_as_uint(x) >> 16; }
__global__ void k_atom_v2_bf16x2_max(uint32_t* c) {
  const int t = TID();
  const float i = (float)(t / kNC);
  const uint32_t w0 = bf16_of(i) | (bf16_of(-(i + 1.0f)) << 16);
  const uint32_t w1 = bf16_of(0.5f * i) | (bf16_of(-(2.0f * i + 2.0f)) << 16);
  uint32_t r0, r1;
  asm volatile("atom.global.v2.bf16x2.max.noftz {%0, %1}, [%2], {%3, %4};"
               : "=r"(r0), "=r"(r1)
               : "l"(c + 2 * (t % kNC)), "r"(w0), "r"(w1));
  (void)r0;
  (void)r1;
}
// f16 elements (i, -i, 256 + i, 1000 - i), all exact.
__global__ void k_atom_v4_f16_min(u16* c) {
  const int t = TID();
  const float i = (float)(t / kNC);
  const u16 e0 = __half_as_ushort(__float2half(i)), e1 = __half_as_ushort(__float2half(-i));
  const u16 e2 = __half_as_ushort(__float2half(256.0f + i));
  const u16 e3 = __half_as_ushort(__float2half(1000.0f - i));
  u16 r0, r1, r2, r3;
  asm volatile("atom.global.v4.f16.min.noftz {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8};"
               : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3)
               : "l"(c + 4 * (t % kNC)), "h"(e0), "h"(e1), "h"(e2), "h"(e3));
  (void)r0;
  (void)r1;
  (void)r2;
  (void)r3;
}
__global__ void k_red_v2_f16x2(uint32_t* c) {
  const int t = TID();
  asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1, %2};" ::"l"(c + 2 * (t % kNC)),
               "r"(0x3C003C00u), "r"(0x40004000u));  // (1, 1), (2, 2)
}

// ---- .cluster scope --------------------------------------------------------------------------
__global__ void k_atom_cluster_add(uint32_t* c, uint32_t* old) {
  const int t = TID();
  uint32_t r;
  asm volatile("atom.relaxed.cluster.global.add.u32 %0, [%1], 1;" : "=r"(r) : "l"(c + t % kNC));
  old[t] = r;
}
__global__ void k_red_cluster_add(uint32_t* c) {
  const int t = TID();
  asm volatile("red.release.cluster.global.add.u32 [%0], 3;" ::"l"(c + t % kNC));
}
// value (i odd ? -i : i) with i = t / kNC; max = 254.
__global__ void k_atom_cluster_max(int* c) {
  const int t = TID();
  const int i = t / kNC;
  int r;
  asm volatile("atom.cluster.global.max.s32 %0, [%1], %2;"
               : "=r"(r)
               : "l"(c + t % kNC), "r"(i & 1 ? -i : i));
  (void)r;
}
__global__ void k_red_cluster_bf16(u16* c) {
  const int t = TID();
  asm volatile("red.relaxed.cluster.global.add.noftz.bf16 [%0], %1;" ::"l"(c + t % kNC),
               "h"((u16)0x3F80));
}

// ---- .shared::cluster (own CTA's shared memory) ---------------------------------------------
// fin: per block kSC words of each of the four counter sets; old: per thread two old values.
__global__ void k_shared_cluster(uint32_t* fin, uint32_t* old) {
  __shared__ uint32_t s_add[kSC], s_red[kSC], s_scoped[kSC];
  __shared__ u16 s_f16[kSC];
  if (threadIdx.x < kSC) {
    s_add[threadIdx.x] = 0;
    s_red[threadIdx.x] = 0;
    s_scoped[threadIdx.x] = 0;
    s_f16[threadIdx.x] = 0;
  }
  __syncthreads();
  const int j = threadIdx.x % kSC;
  const uint32_t a_add = (uint32_t)__cvta_generic_to_shared(&s_add[j]);
  const uint32_t a_red = (uint32_t)__cvta_generic_to_shared(&s_red[j]);
  const uint32_t a_sc = (uint32_t)__cvta_generic_to_shared(&s_scoped[j]);
  const uint32_t a_f16 = (uint32_t)__cvta_generic_to_shared(&s_f16[j]);
  uint32_t r0, r1;
  u16 h;
  asm volatile("atom.shared::cluster.add.u32 %0, [%1], 1;" : "=r"(r0) : "r"(a_add));
  asm volatile("red.shared::cluster.add.u32 [%0], 2;" ::"r"(a_red));
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], 5;" : "=r"(r1) : "r"(a_sc));
  asm volatile("atom.shared::cluster.add.noftz.f16 %0, [%1], %2;"
               : "=h"(h)
               : "r"(a_f16), "h"((u16)0x3C00));
  (void)h;
  old[2 * TID()] = r0;
  old[2 * TID() + 1] = r1;
  __syncthreads();
  if (threadIdx.x < kSC) {
    uint32_t* f = fin + blockIdx.x * 4 * kSC;
    f[threadIdx.x] = s_add[threadIdx.x];
    f[kSC + threadIdx.x] = s_red[threadIdx.x];
    f[2 * kSC + threadIdx.x] = s_scoped[threadIdx.x];
    f[3 * kSC + threadIdx.x] = s_f16[threadIdx.x];
  }
}

// ---- multimem (one replica) -------------------------------------------------------------------
// Loads: thread t reads its own 16-byte slot src + 4 t.
__global__ void k_mm_ld(const uint32_t* src, uint32_t* out) {
  const int t = TID();
  const uint32_t* p = src + 4 * t;
  uint32_t* o = out + 16 * t;
  uint32_t f;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=r"(f) : "l"(p));
  o[0] = f;
  u16 h0, h1, h2, h3;
  asm volatile("multimem.ld_reduce.global.max.acc::f32.v4.bf16 {%0, %1, %2, %3}, [%4];"
               : "=h"(h0), "=h"(h1), "=h"(h2), "=h"(h3)
               : "l"(p));
  o[1] = h0 | ((uint32_t)h1 << 16);
  o[2] = h2 | ((uint32_t)h3 << 16);
  uint32_t w0, w1;
  asm volatile("multimem.ld_reduce.acquire.gpu.global.min.v2.f16x2 {%0, %1}, [%2];"
               : "=r"(w0), "=r"(w1)
               : "l"(p));
  o[3] = w0;
  o[4] = w1;
  u64 b;
  asm volatile("multimem.ld_reduce.relaxed.cluster.global.and.b64 %0, [%1];" : "=l"(b) : "l"(p));
  o[5] = (uint32_t)b;
  o[6] = (uint32_t)(b >> 32);
  uint32_t v0, v1, v2, v3;
  asm volatile("multimem.ld_reduce.weak.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
               : "l"(p));
  o[7] = v0;
  o[8] = v1;
  o[9] = v2;
  o[10] = v3;
}
// Stores: thread t writes its own slots of three destination buffers.
__global__ void k_mm_st(uint32_t* d0, uint32_t* d1, uint32_t* d2) {
  const int t = TID();
  const uint32_t x = 0x01000000u * (t & 0x7F) + (uint32_t)t;
  asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};" ::"l"(d0 + 2 * t), "r"(x),
               "r"(~x));
  const u16 a = (u16)t, b = (u16)(t >> 3) ^ 0x5555, e = (u16)(3 * t), f = (u16)(t * 7 + 1);
  asm volatile(
      "multimem.st.release.gpu.global.v8.bf16 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(d1 +
                                                                                             4 * t),
      "h"(a), "h"(b), "h"(e), "h"(f), "h"(f), "h"(e), "h"(b), "h"(a));
  asm volatile("multimem.st.global.f32 [%0], %1;" ::"l"(d2 + t), "r"(x ^ 0x00FF00FFu));
}
// Reductions on counters (kNC of each).
__global__ void k_mm_red(uint32_t* cu, uint32_t* cf, uint32_t* cb, u16* ch, int* cm) {
  const int t = TID(), k = t % kNC, i = t / kNC;
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(cu + k));
  asm volatile("multimem.red.release.gpu.global.add.f32 [%0], %1;" ::"l"(cf + k), "r"(kSub));
  asm volatile("multimem.red.relaxed.sys.global.add.v2.bf16x2 [%0], {%1, %2};" ::"l"(cb + 2 * k),
               "r"(0x40003F80u), "r"(0x3F803F80u));
  asm volatile("multimem.red.global.add.v4.f16 [%0], {%1, %2, %3, %4};" ::"l"(ch + 4 * k),
               "h"((u16)0x3C00), "h"((u16)0x4000), "h"((u16)0x3C00), "h"((u16)0x4000));
  asm volatile("multimem.red.relaxed.sys.global.min.s32 [%0], %1;" ::"l"(cm + k),
               "r"(i & 1 ? -i : i));
}

// ---- host checks --------------------------------------------------------------------------------

static double bf16_val(uint32_t h) { return (double)u2f((h & 0xFFFFu) << 16); }
// Integer index of an exact multiple of `step`, -1 otherwise.
static long as_index(double v, double step) {
  const double q = v / step;
  return (q >= 0 && q == floor(q)) ? (long)q : -1;
}

// Each counter k (hit by threads k, k + nc, ...) must have returned every partial sum
// 0, step, 2 step, ... exactly once. idx maps an old value to its partial-sum index.
template <class T, class Idx>
void check_olds(Item& it, const std::vector<T>& old, size_t stride, size_t off, int nc, int per,
                Idx idx) {
  const size_t nt = old.size() / stride;
  for (int k = 0; k < nc; ++k) {
    std::vector<int> seen(per, 0);
    bool ok = true;
    u64 bad = 0;
    for (size_t t = k; t < nt; t += nc) {
      const long v = idx(old[t * stride + off]);
      if (v < 0 || v >= per || seen[v]++) {
        if (ok) bad = (u64)old[t * stride + off];
        ok = false;
      }
    }
    it.check(ok, "counter=%d first bad old=%llx", k, bad);
  }
}
template <class T>
void check_final(Item& it, const std::vector<T>& fin, size_t stride, size_t off, int nc, u64 want) {
  for (int k = 0; k < nc; ++k) {
    const u64 g = (u64)fin[k * stride + off];
    it.check(g == want, "counter=%d got=%llx want=%llx", k, g, want);
  }
}

static const uint32_t kBf16_256 = 0x4380u, kBf16_512 = 0x4400u;

int main() {
  // ------------------------------------------------------------ bf16 .noftz add
  {
    Dev<u16> c(kNC), o(kNT);
    k_atom_bf16<<<kBlocks, kThreads>>>(c.p, o.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.add.noftz.bf16");
    check_olds(it, o.get(), 1, 0, kNC, kPer, [](u16 h) { return as_index(bf16_val(h), 1.0); });
    check_final(it, c.get(), 1, 0, kNC, kBf16_256);
  }
  {
    Dev<uint32_t> c(kNC), o(kNT);
    k_atom_bf16x2<<<kBlocks, kThreads>>>(c.p, o.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.add.noftz.bf16x2");
    // Both halves come from one atomic update: hi must be exactly 2 * lo.
    check_olds(it, o.get(), 1, 0, kNC, kPer, [](uint32_t w) {
      const double lo = bf16_val(w), hi = bf16_val(w >> 16);
      return hi == 2.0 * lo ? as_index(lo, 1.0) : -1;
    });
    check_final(it, c.get(), 1, 0, kNC, kBf16_256 | (kBf16_512 << 16));
  }
  {
    Dev<u16> c(kNC);
    Dev<uint32_t> c2(kNC);
    k_red_bf16<<<kBlocks, kThreads>>>(c.p);
    k_red_bf16x2<<<kBlocks, kThreads>>>(c2.p);
    MISC_SYNC();
    {
      Item it("misc.e2e.red.global.add.noftz.bf16");
      check_final(it, c.get(), 1, 0, kNC, kBf16_256);
    }
    {
      Item it("misc.e2e.red.global.add.noftz.bf16x2");
      check_final(it, c2.get(), 1, 0, kNC, kBf16_256 | (kBf16_512 << 16));
    }
  }

  // ------------------------------------------------------------ f32 .noftz add (subnormal)
  {
    auto sub_idx = [](uint32_t b) { return b % kSub == 0 ? (long)(b / kSub) : -1L; };
    struct F {
      const char* name;
      void (*k)(uint32_t*, uint32_t*);
    };
    for (const F& f : {F{"misc.e2e.atom.global.add.noftz.f32", k_atom_f32_global},
                       F{"misc.e2e.atom.add.noftz.f32", k_atom_f32_generic}}) {
      Dev<uint32_t> c(kNC), o(kNT);
      f.k<<<kBlocks, kThreads>>>(c.p, o.p);
      MISC_SYNC();
      Item it(f.name);
      check_olds(it, o.get(), 1, 0, kNC, kPer, sub_idx);
      check_final(it, c.get(), 1, 0, kNC, (u64)kPer * kSub);
    }
    {
      Dev<uint32_t> c(kNC);
      k_red_f32<<<kBlocks, kThreads>>>(c.p);
      MISC_SYNC();
      Item it("misc.e2e.red.global.add.noftz.f32");
      check_final(it, c.get(), 1, 0, kNC, (u64)kPer * kSub);
    }
    {
      Dev<uint32_t> fin(kBlocks * kSC), o(kNT);
      k_atom_f32_shared<<<kBlocks, kThreads>>>(fin.p, o.p);
      MISC_SYNC();
      Item it("misc.e2e.atom.shared.add.noftz.f32");
      const std::vector<uint32_t> ov = o.get();
      for (int b = 0; b < kBlocks; ++b) {
        const std::vector<uint32_t> blk(ov.begin() + b * kThreads, ov.begin() + (b + 1) * kThreads);
        check_olds(it, blk, 1, 0, kSC, kSPer, sub_idx);
      }
      check_final(it, fin.get(), 1, 0, kBlocks * kSC, (u64)kSPer * kSub);
    }
  }

  // ------------------------------------------------------------ vector forms
  {
    Dev<uint32_t> c(2 * kNC), o(2 * kNT);
    k_atom_v2_f32<<<kBlocks, kThreads>>>(c.p, o.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.v2.f32.add");
    const std::vector<uint32_t> ov = o.get(), cv = c.get();
    check_olds(it, ov, 2, 0, kNC, kPer, [](uint32_t b) { return as_index(u2f(b), 1.0); });
    check_olds(it, ov, 2, 1, kNC, kPer, [](uint32_t b) { return as_index(u2f(b), 2.0); });
    check_final(it, cv, 2, 0, kNC, f2u(256.0f));
    check_final(it, cv, 2, 1, kNC, f2u(512.0f));
  }
  {
    Dev<uint32_t> c(4 * kNC);
    k_red_v4_f32<<<kBlocks, kThreads>>>(c.p);
    MISC_SYNC();
    Item it("misc.e2e.red.global.add.noftz.v4.f32");
    const std::vector<uint32_t> cv = c.get();
    for (int j = 0; j < 4; ++j) check_final(it, cv, 4, j, kNC, (u64)(j + 1) * kPer * kSub);
  }
  {
    Dev<u16> c(8 * kNC), o(8 * kNT);
    k_atom_v8_bf16<<<kBlocks, kThreads>>>(c.p, o.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.v8.bf16.add.noftz");
    const std::vector<u16> ov = o.get(), cv = c.get();
    for (int j = 0; j < 8; ++j) {
      check_olds(it, ov, 8, j, kNC, kPer, [](u16 h) { return as_index(bf16_val(h), 1.0); });
      check_final(it, cv, 8, j, kNC, kBf16_256);
    }
  }
  {
    std::vector<uint32_t> init(2 * kNC, 0xFF80FF80u);  // -inf in every half
    Dev<uint32_t> c(init);
    k_atom_v2_bf16x2_max<<<kBlocks, kThreads>>>(c.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.v2.bf16x2.max.noftz");
    const std::vector<uint32_t> cv = c.get();
    check_final(it, cv, 2, 0, kNC, 0x437Fu | (0xBF80u << 16));  // (255, -1)
    check_final(it, cv, 2, 1, kNC, 0x42FFu | (0xC000u << 16));  // (127.5, -2)
  }
  {
    std::vector<u16> init(4 * kNC, 0x7C00);  // +inf
    Dev<u16> c(init);
    k_atom_v4_f16_min<<<kBlocks, kThreads>>>(c.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.v4.f16.min.noftz");
    const std::vector<u16> cv = c.get();
    const u64 want[4] = {0x0000u, 0xDBF8u, 0x5C00u, 0x61D2u};  // 0, -255, 256, 745
    for (int j = 0; j < 4; ++j) check_final(it, cv, 4, j, kNC, want[j]);
  }
  {
    Dev<uint32_t> c(2 * kNC);
    k_red_v2_f16x2<<<kBlocks, kThreads>>>(c.p);
    MISC_SYNC();
    Item it("misc.e2e.red.global.add.noftz.v2.f16x2");
    const std::vector<uint32_t> cv = c.get();
    check_final(it, cv, 2, 0, kNC, 0x5C005C00u);  // (256, 256)
    check_final(it, cv, 2, 1, kNC, 0x60006000u);  // (512, 512)
  }

  // ------------------------------------------------------------ .cluster scope
  {
    Dev<uint32_t> c(kNC), o(kNT), c3(kNC);
    std::vector<int> init(kNC, (int)0x80000000);
    Dev<int> cm(init);
    Dev<u16> cb(kNC);
    k_atom_cluster_add<<<kBlocks, kThreads>>>(c.p, o.p);
    k_red_cluster_add<<<kBlocks, kThreads>>>(c3.p);
    k_atom_cluster_max<<<kBlocks, kThreads>>>(cm.p);
    k_red_cluster_bf16<<<kBlocks, kThreads>>>(cb.p);
    MISC_SYNC();
    {
      Item it("misc.e2e.atom.relaxed.cluster.global.add.u32");
      check_olds(it, o.get(), 1, 0, kNC, kPer, [](uint32_t v) { return (long)v; });
      check_final(it, c.get(), 1, 0, kNC, kPer);
    }
    {
      Item it("misc.e2e.red.release.cluster.global.add.u32");
      check_final(it, c3.get(), 1, 0, kNC, 3 * kPer);
    }
    {
      Item it("misc.e2e.atom.cluster.global.max.s32");
      check_final(it, cm.get(), 1, 0, kNC, 254);
    }
    {
      Item it("misc.e2e.red.relaxed.cluster.global.add.noftz.bf16");
      check_final(it, cb.get(), 1, 0, kNC, kBf16_256);
    }
  }

  // ------------------------------------------------------------ .shared::cluster
  {
    Dev<uint32_t> fin(kBlocks * 4 * kSC), o(2 * kNT);
    k_shared_cluster<<<kBlocks, kThreads>>>(fin.p, o.p);
    MISC_SYNC();
    const std::vector<uint32_t> fv = fin.get(), ov = o.get();
    Item it("misc.e2e.atom.shared::cluster");
    for (int b = 0; b < kBlocks; ++b) {
      const std::vector<uint32_t> blk(ov.begin() + 2 * b * kThreads,
                                      ov.begin() + 2 * (b + 1) * kThreads);
      check_olds(it, blk, 2, 0, kSC, kSPer, [](uint32_t v) { return (long)v; });
      check_olds(it, blk, 2, 1, kSC, kSPer, [](uint32_t v) { return v % 5 ? -1L : (long)(v / 5); });
      const std::vector<uint32_t> f(fv.begin() + 4 * kSC * b, fv.begin() + 4 * kSC * (b + 1));
      for (int j = 0; j < kSC; ++j) {
        it.check(f[j] == kSPer, "block=%d add counter=%d got=%x", b, j, f[j]);
        it.check(f[kSC + j] == 2 * kSPer, "block=%d red counter=%d got=%x", b, j, f[kSC + j]);
        it.check(f[2 * kSC + j] == 5 * kSPer, "block=%d scoped counter=%d got=%x", b, j,
                 f[2 * kSC + j]);
        it.check(f[3 * kSC + j] == 0x5000u, "block=%d f16 counter=%d got=%x", b, j,
                 f[3 * kSC + j]);  // 32.0
      }
    }
  }

  // ------------------------------------------------------------ multimem
  {
    std::vector<uint32_t> src(4 * kNT);
    Rng r(0x3317);
    for (auto& x : src) x = r.u32();
    Dev<uint32_t> ds(src), dout(16 * kNT);
    k_mm_ld<<<kBlocks, kThreads>>>(ds.p, dout.p);
    MISC_SYNC();
    const std::vector<uint32_t> out = dout.get();
    Item it("misc.e2e.multimem.ld_reduce");
    for (int t = 0; t < kNT; ++t) {
      const uint32_t* s = &src[4 * t];
      const uint32_t* o = &out[16 * t];
      const bool ok = o[0] == s[0] && o[1] == s[0] && o[2] == s[1] && o[3] == s[0] &&
                      o[4] == s[1] && o[5] == s[0] && o[6] == s[1] && o[7] == s[0] &&
                      o[8] == s[1] && o[9] == s[2] && o[10] == s[3];
      it.check(ok, "t=%d src=%x,%x,%x,%x out=%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x", t, s[0], s[1], s[2],
               s[3], o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10]);
    }
  }
  {
    Dev<uint32_t> d0(2 * kNT), d1(4 * kNT), d2(kNT);
    k_mm_st<<<kBlocks, kThreads>>>(d0.p, d1.p, d2.p);
    MISC_SYNC();
    const std::vector<uint32_t> v0 = d0.get(), v1 = d1.get(), v2 = d2.get();
    Item it("misc.e2e.multimem.st");
    for (int t = 0; t < kNT; ++t) {
      const uint32_t x = 0x01000000u * (t & 0x7F) + (uint32_t)t;
      const uint32_t a = (u16)t, b = (u16)((u16)(t >> 3) ^ 0x5555), e = (u16)(3 * t),
                     f = (u16)(t * 7 + 1);
      const bool ok = v0[2 * t] == x && v0[2 * t + 1] == ~x && v1[4 * t] == (a | (b << 16)) &&
                      v1[4 * t + 1] == (e | (f << 16)) && v1[4 * t + 2] == (f | (e << 16)) &&
                      v1[4 * t + 3] == (b | (a << 16)) && v2[t] == (x ^ 0x00FF00FFu);
      it.check(ok, "t=%d", t);
    }
  }
  {
    Dev<uint32_t> cu(kNC), cf(kNC), cb(2 * kNC);
    Dev<u16> ch(4 * kNC);
    std::vector<int> init(kNC, 0x7FFFFFFF);
    Dev<int> cm(init);
    k_mm_red<<<kBlocks, kThreads>>>(cu.p, cf.p, cb.p, ch.p, cm.p);
    MISC_SYNC();
    Item it("misc.e2e.multimem.red");
    check_final(it, cu.get(), 1, 0, kNC, kPer);
    check_final(it, cf.get(), 1, 0, kNC, (u64)kPer * kSub);
    const std::vector<uint32_t> bv = cb.get();
    check_final(it, bv, 2, 0, kNC, kBf16_256 | (kBf16_512 << 16));
    check_final(it, bv, 2, 1, kNC, kBf16_256 | (kBf16_256 << 16));
    const std::vector<u16> hv = ch.get();
    for (int j = 0; j < 4; ++j) check_final(it, hv, 4, j, kNC, j & 1 ? 0x6000u : 0x5C00u);
    check_final(it, cm.get(), 1, 0, kNC, (u64)(long long)-255);
  }
  return finish();
}
