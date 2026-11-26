// mbarrier end-to-end test (sm_90a PTX, translated by tools/ptxemu/rules_async.py).
//
// k_phases: 128 threads exchange data through a double buffer guarded by one mbarrier for
// kRounds phases, cycling through four arrive / wait forms (state, generic address, parity with
// .cluster scope, relaxed + sync_restrict fences).
// k_seq: one thread walks a fixed sequence of mbarrier operations (expect_tx / complete_tx,
// noComplete + pending_count, arrive_drop, try_wait time hint, inval / re-init, layout check,
// report operands) and records every predicate; the expected values follow from the PTX ISA
// rules (ptx.txt 21043-21050 phase completion, 21853-21866 pending_count, 21660-21710 waits).
//
// WGEMU-COVERS-FEATURE: mbarrier_arrive_count_without_noComplete
// WGEMU-COVERS-FEATURE: mbarrier_wait_report_operands
#include "async_common.cuh"

constexpr int kThreads = 128;
constexpr int kRounds = 8;

__device__ __forceinline__ uint32_t tw_parity(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok;
}

__global__ void k_phases(uint32_t* data_out, uint32_t* parity_out) {
  __shared__ alignas(8) uint64_t bar;
  __shared__ uint32_t buf[2][kThreads];
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  if (t == 0) {
    mbar_init(b, kThreads);
    fence_mbar_init();
  }
  __syncthreads();
  for (int r = 0; r < kRounds; ++r) {
    buf[r & 1][t] = (uint32_t)r * 1000u + t;
    const uint32_t par = (uint32_t)r & 1u;
    uint64_t st = 0;
    uint32_t done = 0;
    switch (r & 3) {
      case 0:  // state form, .shared::cta, release.cta / test_wait
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
                     : "=l"(st)
                     : "r"(b)
                     : "memory");
        while (!done) {
          asm volatile(
              "{\n\t.reg .pred p;\n\t"
              "mbarrier.test_wait.shared::cta.b64 p, [%1], %2;\n\t"
              "selp.u32 %0, 1, 0, p;\n\t}"
              : "=r"(done)
              : "r"(b), "l"(st)
              : "memory");
        }
        break;
      case 1:  // generic address arrive, try_wait with the state
        asm volatile("mbarrier.arrive.b64 %0, [%1];" : "=l"(st) : "l"(&bar) : "memory");
        while (!done) {
          asm volatile(
              "{\n\t.reg .pred p;\n\t"
              "mbarrier.try_wait.shared::cta.b64 p, [%1], %2;\n\t"
              "selp.u32 %0, 1, 0, p;\n\t}"
              : "=r"(done)
              : "r"(b), "l"(st)
              : "memory");
        }
        break;
      case 2:  // .shared::cluster arrive (own CTA, cluster of one), parity wait at .cluster scope
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(b)
                     : "memory");
        while (!done) {
          asm volatile(
              "{\n\t.reg .pred p;\n\t"
              "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
              "selp.u32 %0, 1, 0, p;\n\t}"
              : "=r"(done)
              : "r"(b), "r"(par)
              : "memory");
        }
        break;
      default:  // relaxed arrive with an explicit count, fences give the ordering (ptx.txt 19780)
        asm volatile(
            "fence.release.sync_restrict::shared::cta.cluster;\n\t"
            "mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1], 1;"
            : "=l"(st)
            : "r"(b)
            : "memory");
        while (!done) {
          asm volatile(
              "{\n\t.reg .pred p;\n\t"
              "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], %2;\n\t"
              "selp.u32 %0, 1, 0, p;\n\t}"
              : "=r"(done)
              : "r"(b), "r"(par)
              : "memory");
        }
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
        break;
    }
    data_out[r * kThreads + t] = buf[r & 1][(t + 37u) % kThreads];
    if (t == 0) {
      // Phase r is complete and phase r + 1 cannot complete without this thread's next arrive.
      parity_out[2 * r + 0] = tw_parity(b, par);
      parity_out[2 * r + 1] = tw_parity(b, par ^ 1u);
    }
  }
}

constexpr int kSeqOut = 26;

__global__ void k_seq(uint32_t* o) {
  __shared__ alignas(8) uint64_t bars[3];
  const uint32_t b0 = smem_u32(&bars[0]);
  const uint32_t b1 = smem_u32(&bars[1]);
  const uint32_t b2 = smem_u32(&bars[2]);
  uint64_t s1, s2, s3, s4;
  uint32_t v;

  // A: arrive.expect_tx then two complete_tx halves.
  mbar_init(b0, 1);
  fence_mbar_init();
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 256;"
               : "=l"(s3)
               : "r"(b0)
               : "memory");
  o[0] = tw_parity(b0, 0);
  asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], 128;" ::"r"(b0) : "memory");
  o[1] = tw_parity(b0, 0);
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" ::"r"(b0), "r"(128) : "memory");
  o[2] = tw_parity(b0, 0);

  // B: expect_tx without arrive, then the arrive, then the bytes (phase 1).
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(b0), "r"(64)
               : "memory");
  o[3] = tw_parity(b0, 1);
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];"
               : "=l"(s3)
               : "r"(b0)
               : "memory");
  o[4] = tw_parity(b0, 1);
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], 64;" ::"r"(b0) : "memory");
  o[5] = tw_parity(b0, 1);

  // C: noComplete arrivals and pending_count (count before the arrive, ptx.txt 21862-21864).
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(b1) : "memory");
  asm volatile("mbarrier.arrive.noComplete.shared::cta.b64 %0, [%1], 1;"
               : "=l"(s1)
               : "r"(b1)
               : "memory");
  asm volatile("mbarrier.arrive.noComplete.release.cta.shared::cta.b64 %0, [%1], %2;"
               : "=l"(s2)
               : "r"(b1), "r"(2)
               : "memory");
  asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(v) : "l"(s1));
  o[6] = v;
  asm volatile("mbarrier.pending_count.layout::v0.b64 %0, %1;" : "=r"(v) : "l"(s2));
  o[7] = v;
  o[8] = tw_parity(b1, 0);
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], 1;" : "=l"(s3) : "r"(b1) : "memory");
  o[9] = tw_parity(b1, 0);
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b1), "l"(s2)
      : "memory");
  o[10] = v;

  // D: arrive_drop lowers the expected count for the following phases too.
  mbar_init(b2, 2);
  asm volatile("mbarrier.arrive_drop.shared::cta.b64 _, [%0];" ::"r"(b2) : "memory");
  o[11] = tw_parity(b2, 0);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b2) : "memory");
  o[12] = tw_parity(b2, 0);
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(b2) : "memory");
  o[13] = tw_parity(b2, 1);

  // E: try_wait with a time hint on the incomplete current phase (2) and the completed phase 1.
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2), "r"(0), "r"(1000)
      : "memory");
  o[14] = v;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 1, 1000;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2)
      : "memory");
  o[15] = v;

  // F: inval, re-init with an explicit layout, layout checks.
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(b2) : "memory");
  asm volatile("mbarrier.init.layout::v0.shared::cta.b64 [%0], 1;" ::"r"(b2) : "memory");
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.check_layout.layout::v0.shared::cta.b64 p, [%1];\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2)
      : "memory");
  o[16] = v;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.check_layout.layout::v1.shared::cta.b64 p, [%1];\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2)
      : "memory");
  o[17] = v;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(s4) : "r"(b2) : "memory");
  o[18] = tw_parity(b2, 0);

  // G: report operands are zero for .layout::v0 (ptx.txt 21690-21691); phase 0 is complete.
  {
    uint32_t wc, rp, rv;
    asm volatile(
        "{\n\t.reg .pred p, q;\n\t.reg .b8 rv;\n\t"
        "mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 p|q, rv, [%3], 0;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "selp.u32 %1, 1, 0, q;\n\t"
        "cvt.u32.u8 %2, rv;\n\t}"
        : "=r"(wc), "=r"(rp), "=r"(rv)
        : "r"(b2)
        : "memory");
    o[19] = wc;
    o[20] = rp;
    o[21] = rv;
  }
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.phase_type::primary.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2), "l"(s4)
      : "memory");
  o[22] = v;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.phase_type::conditional.shared::cta.b64 p, [%1], 0;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b2)
      : "memory");
  o[23] = v;

  // H: .sem.scope forms on the completed phase 0 of b1.
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b1)
      : "memory");
  o[24] = v;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.relaxed.cluster.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(v)
      : "r"(b1), "l"(s1)
      : "memory");
  o[25] = v;
}

int main() {
  uint32_t *d_data, *d_par, *d_seq;
  WG_CHECK(cudaMalloc(&d_data, sizeof(uint32_t) * kRounds * kThreads));
  WG_CHECK(cudaMalloc(&d_par, sizeof(uint32_t) * 2 * kRounds));
  WG_CHECK(cudaMalloc(&d_seq, sizeof(uint32_t) * kSeqOut));
  WG_CHECK(cudaMemset(d_data, 0xFF, sizeof(uint32_t) * kRounds * kThreads));
  WG_CHECK(cudaMemset(d_par, 0xFF, sizeof(uint32_t) * 2 * kRounds));
  WG_CHECK(cudaMemset(d_seq, 0xFF, sizeof(uint32_t) * kSeqOut));

  k_phases<<<1, kThreads>>>(d_data, d_par);
  WG_CHECK(cudaGetLastError());
  k_seq<<<1, 1>>>(d_seq);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> data(kRounds * kThreads), par(2 * kRounds), seq(kSeqOut);
  WG_CHECK(cudaMemcpy(data.data(), d_data, data.size() * 4, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(par.data(), d_par, par.size() * 4, cudaMemcpyDeviceToHost));
  WG_CHECK(cudaMemcpy(seq.data(), d_seq, seq.size() * 4, cudaMemcpyDeviceToHost));

  Checker phases("async.mbarrier.phases");
  for (int r = 0; r < kRounds; ++r) {
    for (int t = 0; t < kThreads; ++t) {
      phases.eq(r * kThreads + t, data[r * kThreads + t],
                (uint32_t)r * 1000u + (uint32_t)((t + 37) % kThreads));
    }
    phases.eq(100000 + 2 * r, par[2 * r], 1);
    phases.eq(100000 + 2 * r + 1, par[2 * r + 1], 0);
  }
  wg_report(phases);

  static const uint32_t want[kSeqOut] = {0, 0, 1, 0, 0, 1,        // A, B
                                         4, 3, 0, 1, 1,           // C
                                         0, 1, 1, 0, 1,           // D, E
                                         1, 0, 1, 1, 0, 0, 1, 1,  // F, G
                                         1, 1};                   // H
  Checker sq("async.mbarrier.sequence");
  for (int i = 0; i < kSeqOut; ++i) sq.eq(i, seq[i], want[i]);
  wg_report(sq);

  WG_CHECK(cudaFree(d_data));
  WG_CHECK(cudaFree(d_par));
  WG_CHECK(cudaFree(d_seq));
  return wg_finish();
}
