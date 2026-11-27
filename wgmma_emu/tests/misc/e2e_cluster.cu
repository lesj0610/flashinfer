// End-to-end check of the sm_90 cluster-level instructions that tools/ptxemu/rules_misc.py
// maps onto an implicit cluster of one CTA (fork-only). Owner: T2. Built for sm_90a by
// tools/nvcc_emu. Launches here are plain (implicit 1x1x1 clusters, ptx.txt 900).
//
// Covered: barrier.cluster.arrive{.release,.relaxed}{.aligned} / .wait{.acquire}{.aligned}
// (software split barrier, early-exiting threads leave it: "wait for all non-exited threads",
// ptx.txt 19580); %clusterid, %nclusterid, %cluster_ctaid, %cluster_nctaid, %cluster_ctarank,
// %cluster_nctarank, %aggr_smem_size; mapa, getctarank,
// ld/st .shared::cluster; elect.sync; setmaxnreg and griddepcontrol (no-ops).
//
// Expectations for an implicit launch: %clusterid = %ctaid, %nclusterid = %nctaid,
// %cluster_ctaid = 0, %cluster_nctaid = 1, %cluster_ctarank = 0, %cluster_nctarank = 1,
// fourth vector element = 0 (ptx.txt 34879). %is_explicit_cluster needs Ctx.cluster_info (T1)
// and is tested in e2e_cluster_dims.cu / e2e_block_size.cu.
// [inference] %aggr_smem_size = %total_smem_size + %reserved_smem_offset_cap (ptx.txt 35255
// describes the sum of the user allocation and the reserved region).
// mapa results are checked through the accesses they enable, not as numbers: a shared::cluster
// address of the own CTA need not equal its shared::cta address.
#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kPhases = 8;

__device__ __forceinline__ int flat_tid() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

// ---- barrier.cluster ---------------------------------------------------------------------------

// Each phase: every active thread writes its slot, passes the barrier, reads another thread's
// slot, then passes a second barrier before the next phase overwrites the slots.
// variant 0: default qualifiers, the first arrive split by thread parity (non-aligned, one
// warp arrives in two groups).
// variant 1: .release.aligned / .acquire.aligned, then fence.acq_rel.cta + .relaxed.aligned.
// variant 2: .release / .acquire with independent work between arrive and wait.
// Threads with flat id >= active exit before the first barrier.
template <int kVariant>
__global__ void k_cbar(int active, uint32_t* bad) {
  extern __shared__ uint32_t slot[];
  const int t = flat_tid(), b = blockIdx.x;
  if (t >= active) return;
  uint32_t nbad = 0, work = (uint32_t)t;
  for (int ph = 0; ph < kPhases; ++ph) {
    slot[t] = 0x10000u * (uint32_t)(ph + 1) + (uint32_t)t + 0x100u * (uint32_t)b;
    if (kVariant == 0) {
      // Two different strings so that the compiler cannot merge the branches.
      if (threadIdx.x & 1)
        asm volatile("barrier.cluster.arrive;" ::: "memory");
      else
        asm volatile("barrier.cluster.arrive.release;" ::: "memory");
      asm volatile("barrier.cluster.wait;" ::: "memory");
    } else if (kVariant == 1) {
      asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
      asm volatile("barrier.cluster.arrive.release;" ::: "memory");
      for (int i = 0; i < 64; ++i) work = work * 1664525u + 1013904223u;
      asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
    }
    const int o = (t * 7 + 3 + ph) % active;
    const uint32_t want = 0x10000u * (uint32_t)(ph + 1) + (uint32_t)o + 0x100u * (uint32_t)b;
    nbad += slot[o] != want;
    if (kVariant == 1) {
      __threadfence_block();
      asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    } else {
      asm volatile("barrier.cluster.arrive;" ::: "memory");
      asm volatile("barrier.cluster.wait;" ::: "memory");
    }
  }
  bad[b * 1024 + t] = nbad + (work == 0xFFFFFFFFu);  // keep work alive
}

// ---- special registers
// ---------------------------------------------------------------------------

constexpr int kSregs = 20;

__global__ void k_sregs(uint32_t* out) {
  __shared__ uint32_t pad[1250];  // 5000 bytes of static shared memory
  extern __shared__ uint32_t dyn[];
  const int cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  uint32_t* o = out + kSregs * (cta * blockDim.x + threadIdx.x);
  uint32_t v[4];
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(o[0]));
  asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(o[1]));
  asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(o[2]));
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(o[3]));
  asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(o[4]));
  asm volatile("mov.u32 %0, %%nclusterid.z;" : "=r"(o[5]));
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(o[6]));
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(o[7]));
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(o[8]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(o[9]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(o[10]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(o[11]));
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(o[12]));
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(o[13]));
  asm volatile("mov.v4.u32 {%0, %1, %2, %3}, %%clusterid;"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]));
  o[14] = v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24);
  asm volatile("mov.v4.u32 {%0, %1, %2, %3}, %%cluster_nctaid;"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]));
  o[15] = v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24);
  asm volatile("mov.u32 %0, %%aggr_smem_size;" : "=r"(o[16]));
  asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(o[17]));
  asm volatile("mov.u32 %0, %%reserved_smem_offset_cap;" : "=r"(o[18]));
  // Touch both shared arrays so that they are allocated.
  pad[threadIdx.x] = threadIdx.x;
  dyn[threadIdx.x] = threadIdx.x;
  __syncthreads();
  o[19] = pad[(threadIdx.x + 1) % blockDim.x] + dyn[(threadIdx.x + 2) % blockDim.x];
}

// ---- mapa / getctarank / ld, st .shared::cluster -----------------------------------------------

constexpr int kMapaOut = 8;

__global__ void k_mapa(uint32_t* out) {
  __shared__ uint32_t s[4 * 128];
  const int t = threadIdx.x;
  uint32_t* o = out + kMapaOut * (blockIdx.x * blockDim.x + t);
  for (int i = t; i < 4 * 128; i += blockDim.x) s[i] = 0;
  __syncthreads();
  const uint32_t a0 = (uint32_t)__cvta_generic_to_shared(&s[t]);
  const uint32_t a1 = (uint32_t)__cvta_generic_to_shared(&s[128 + t]);
  const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(&s[256 + t]);
  uint32_t rank, m0, m1, k0, k1, x;
  uint64_t m2, g3;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(m0) : "r"(a0), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(m1) : "r"(a1));
  asm volatile("mapa.shared::cluster.u64 %0, %1, 0;" : "=l"(m2) : "l"((uint64_t)a2));
  asm volatile("mapa.u64 %0, %1, %2;" : "=l"(g3) : "l"(&s[384 + t]), "r"(rank));
  const uint32_t v = 0xC0DE0000u + 0x1000u * blockIdx.x + (uint32_t)t;
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(m0), "r"(v) : "memory");
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(m1), "r"(v + 1) : "memory");
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"l"(m2), "r"(v + 2) : "memory");
  *(volatile uint32_t*)g3 = v + 3;
  asm volatile("getctarank.shared::cluster.u32 %0, %1;" : "=r"(k0) : "r"(a0));
  asm volatile("getctarank.u32 %0, %1;" : "=r"(k1) : "l"(&s[t]));
  __syncthreads();
  o[0] = s[t];
  o[1] = s[128 + t];
  o[2] = s[256 + t];
  o[3] = s[384 + t];
  // ld .shared::cluster of another thread's slot through the shared::cta address.
  const uint32_t a4 = (uint32_t)__cvta_generic_to_shared(&s[(t + 5) % blockDim.x]);
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(x) : "r"(a4) : "memory");
  o[4] = x;
  o[5] = k0;
  o[6] = k1;
  o[7] = rank;
}

// ---- elect.sync
// ----------------------------------------------------------------------------------

// A function, not a constexpr array: device code may not index a host constexpr array at run
// time.
constexpr int kNMasks = 5;
__host__ __device__ constexpr uint32_t elect_mask(int m) {
  return m == 0   ? 0xFFFFFFFFu
         : m == 1 ? 0x0000FF00u
         : m == 2 ? 0x80000001u
         : m == 3 ? 0x55555555u
                  : 0x00000010u;
}

// out[(set * 32 + lane) * 2 + {0, 1}] = (d, p); lanes outside the mask leave 0xFFFFFFFF.
// set 2 m: immediate mask m; set 2 m + 1: register mask, sink '_' for d.
__global__ void k_elect(uint32_t* out) {
  const uint32_t lane = threadIdx.x;
  for (int m = 0; m < kNMasks; ++m) {
    const uint32_t mask = elect_mask(m);
    if (!((mask >> lane) & 1u)) continue;
    uint32_t d = 0xFFFFFFFFu, p = 0xFFFFFFFFu;
    switch (m) {
#define ELECT_IMM(M, VAL)                                                             \
  case M:                                                                             \
    asm volatile("{ .reg .pred p; elect.sync %0|p, " #VAL "; selp.u32 %1, 1, 0, p; }" \
                 : "=r"(d), "=r"(p));                                                 \
    break;
      ELECT_IMM(0, 0xFFFFFFFF)
      ELECT_IMM(1, 0x0000FF00)
      ELECT_IMM(2, 0x80000001)
      ELECT_IMM(3, 0x55555555)
      ELECT_IMM(4, 0x00000010)
#undef ELECT_IMM
    }
    out[((2 * m) * 32 + lane) * 2] = d;
    out[((2 * m) * 32 + lane) * 2 + 1] = p;
    asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }"
                 : "=r"(p)
                 : "r"(mask));
    out[((2 * m + 1) * 32 + lane) * 2] = 0;
    out[((2 * m + 1) * 32 + lane) * 2 + 1] = p;
  }
}

// ---- setmaxnreg / griddepcontrol
// -------------------------------------------------------------------

__global__ void k_noops(const uint32_t* in, uint32_t* out) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t v = in[i] * 3u + 1u;
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  out[i] = v;
  asm volatile("griddepcontrol.launch_dependents;");
}

// ---- host side ---------------------------------------------------------------------------------

template <int kVariant>
static void run_cbar(const char* name, dim3 block, int active) {
  const int blocks = 6, nt = (int)(block.x * block.y * block.z);
  Dev<uint32_t> bad((size_t)blocks * 1024);
  std::vector<uint32_t> init((size_t)blocks * 1024, 0xFFFFFFFFu);
  bad.put(init);
  k_cbar<kVariant><<<blocks, block, nt * sizeof(uint32_t)>>>(active, bad.p);
  MISC_SYNC();
  const std::vector<uint32_t> bv = bad.get();
  Item it(name);
  for (int b = 0; b < blocks; ++b) {
    for (int t = 0; t < nt; ++t) {
      const uint32_t want = t < active ? 0u : 0xFFFFFFFFu;  // exited threads write nothing
      it.check(bv[b * 1024 + t] == want, "block=%d thread=%d got=%u want=%u", b, t,
               bv[b * 1024 + t], want);
    }
  }
}

static void run_sregs() {
  const dim3 grid(3, 2, 2), block(64);
  const int ncta = 12, dyn_bytes = 3000;
  Dev<uint32_t> out((size_t)kSregs * ncta * block.x);
  k_sregs<<<grid, block, dyn_bytes>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it("misc.e2e.sreg.cluster");
  Item ia("misc.e2e.sreg.aggr_smem_size");
  for (int cz = 0; cz < 2; ++cz) {
    for (int cy = 0; cy < 2; ++cy) {
      for (int cx = 0; cx < 3; ++cx) {
        const int cta = (cz * 2 + cy) * 3 + cx;
        for (int t = 0; t < (int)block.x; ++t) {
          const uint32_t* v = &o[kSregs * (cta * block.x + t)];
          const uint32_t want[16] = {(uint32_t)cx,
                                     (uint32_t)cy,
                                     (uint32_t)cz,
                                     3,
                                     2,
                                     2,
                                     0,
                                     0,
                                     0,
                                     1,
                                     1,
                                     1,
                                     0,
                                     1,
                                     (uint32_t)cx | ((uint32_t)cy << 8) | ((uint32_t)cz << 16),
                                     1u | (1u << 8) | (1u << 16)};
          for (int k = 0; k < 16; ++k) {
            it.check(v[k] == want[k], "cta=(%d,%d,%d) thread=%d field=%d got=%u want=%u", cx, cy,
                     cz, t, k, v[k], want[k]);
          }
          ia.check(v[16] == v[17] + v[18] && v[17] >= 5000u + (uint32_t)dyn_bytes,
                   "cta=%d thread=%d aggr=%u total=%u cap=%u", cta, t, v[16], v[17], v[18]);
        }
      }
    }
  }
}

static void run_mapa() {
  const int blocks = 4, threads = 128;
  Dev<uint32_t> out((size_t)kMapaOut * blocks * threads);
  k_mapa<<<blocks, threads>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item im("misc.e2e.mapa");
  Item ig("misc.e2e.getctarank");
  Item il("misc.e2e.ld_st.shared::cluster");
  for (int b = 0; b < blocks; ++b) {
    for (int t = 0; t < threads; ++t) {
      const uint32_t* v = &o[kMapaOut * (b * threads + t)];
      const uint32_t base = 0xC0DE0000u + 0x1000u * b;
      for (int k = 0; k < 4; ++k) {
        im.check(v[k] == base + t + k, "block=%d thread=%d form=%d got=%08x want=%08x", b, t, k,
                 v[k], base + t + k);
      }
      const uint32_t want4 = base + (uint32_t)((t + 5) % threads);
      il.check(v[4] == want4, "block=%d thread=%d got=%08x want=%08x", b, t, v[4], want4);
      ig.check(v[5] == 0 && v[6] == 0 && v[7] == 0, "block=%d thread=%d ranks=%u,%u,%u", b, t, v[5],
               v[6], v[7]);
    }
  }
}

static void run_elect() {
  Dev<uint32_t> out((size_t)2 * kNMasks * 32 * 2);
  std::vector<uint32_t> init((size_t)2 * kNMasks * 32 * 2, 0xFFFFFFFFu);
  out.put(init);
  k_elect<<<1, 32>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it("misc.e2e.elect.sync");
  for (int set = 0; set < 2 * kNMasks; ++set) {
    const uint32_t mask = elect_mask(set / 2);
    const bool has_d = set % 2 == 0;
    int leaders = 0, leader = -1;
    for (int lane = 0; lane < 32; ++lane) {
      const uint32_t p = o[(set * 32 + lane) * 2 + 1];
      if (!((mask >> lane) & 1u)) {
        it.check(p == 0xFFFFFFFFu, "set=%d lane=%d outside the mask wrote p=%u", set, lane, p);
        continue;
      }
      it.check(p <= 1u, "set=%d lane=%d p=%u", set, lane, p);
      if (p == 1u) {
        ++leaders;
        leader = lane;
      }
    }
    it.check(leaders == 1, "set=%d mask=%08x leaders=%d", set, mask, leaders);
    if (!has_d || leaders != 1) continue;
    for (int lane = 0; lane < 32; ++lane) {
      if (!((mask >> lane) & 1u)) continue;
      const uint32_t d = o[(set * 32 + lane) * 2];
      it.check(d == (uint32_t)leader, "set=%d lane=%d d=%u leader=%d", set, lane, d, leader);
    }
  }
}

static void run_noops() {
  const int n = 4096;
  std::vector<uint32_t> in(n);
  for (int i = 0; i < n; ++i) in[i] = (uint32_t)i * 2654435761u;
  Dev<uint32_t> din(in), dout(n);
  k_noops<<<n / 128, 128>>>(din.p, dout.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = dout.get();
  Item it("misc.e2e.setmaxnreg_griddepcontrol");
  for (int i = 0; i < n; ++i) {
    it.check(o[i] == in[i] * 3u + 1u, "i=%d got=%u want=%u", i, o[i], in[i] * 3u + 1u);
  }
}

int main() {
  run_cbar<0>("misc.e2e.barrier.cluster.default", dim3(256), 256);
  run_cbar<1>("misc.e2e.barrier.cluster.aligned", dim3(8, 4, 4), 128);
  run_cbar<2>("misc.e2e.barrier.cluster.split", dim3(96, 2), 192);
  // Threads 200.. exit first: warp 6 partially, warp 7 entirely.
  run_cbar<0>("misc.e2e.barrier.cluster.exit", dim3(256), 200);
  run_sregs();
  run_mapa();
  run_elect();
  run_noops();
  return finish();
}
