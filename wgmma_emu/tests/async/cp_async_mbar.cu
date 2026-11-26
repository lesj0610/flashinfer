// cp.async.mbarrier.arrive end-to-end test. Every thread issues two 16-byte cp.async copies and
// signals an mbarrier through cp.async.mbarrier.arrive; all threads then wait on the barrier and
// read other threads' copied data (ptx.txt 21560-21632: the arrive-on is triggered after all prior
// cp.async of the thread complete; .noinc does not increment the pending count first).
//
// mode 0: barrier count 128, .noinc async arrivals only.
// mode 1: barrier count 128, cp.async.mbarrier.arrive without .noinc (net zero on the phase)
//         followed by a regular arrive of every thread.
// mode 2: as mode 1 with the generic-address form of both instructions.
#include "async_common.cuh"

constexpr int kThreads = 128;

__global__ void k_cp_async(const uint4* __restrict__ g, uint4* __restrict__ out, int mode) {
  __shared__ alignas(16) uint4 buf[2][kThreads];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  if (t == 0) {
    mbar_init(b, kThreads);
    fence_mbar_init();
  }
  __syncthreads();
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(&buf[0][t])), "l"(g + t)
               : "memory");
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(smem_u32(&buf[1][t])),
               "l"(g + kThreads + t)
               : "memory");
  if (mode == 0) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory");
  } else if (mode == 1) {
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(b) : "memory");
    mbar_arrive(b);
  } else {
    asm volatile("cp.async.mbarrier.arrive.b64 [%0];" ::"l"(&bar) : "memory");
    uint64_t st;
    asm volatile("mbarrier.arrive.b64 %0, [%1];" : "=l"(st) : "l"(&bar) : "memory");
  }
  mbar_wait(b, 0);
  out[t] = buf[0][(t + 5u) % kThreads];
  out[kThreads + t] = buf[1][(t * 3u) % kThreads];
}

int main() {
  std::vector<uint4> h(2 * kThreads);
  for (int i = 0; i < 2 * kThreads; ++i) {
    h[i] = make_uint4(wg_hash32(4 * i), wg_hash32(4 * i + 1), wg_hash32(4 * i + 2),
                      wg_hash32(4 * i + 3));
  }
  uint4 *d_in, *d_out;
  WG_CHECK(cudaMalloc(&d_in, sizeof(uint4) * 2 * kThreads));
  WG_CHECK(cudaMalloc(&d_out, sizeof(uint4) * 2 * kThreads));
  WG_CHECK(cudaMemcpy(d_in, h.data(), sizeof(uint4) * 2 * kThreads, cudaMemcpyHostToDevice));

  static const char* names[3] = {"async.cp_async_mbarrier.noinc",
                                 "async.cp_async_mbarrier.inc_plus_arrive",
                                 "async.cp_async_mbarrier.generic"};
  for (int mode = 0; mode < 3; ++mode) {
    WG_CHECK(cudaMemset(d_out, 0, sizeof(uint4) * 2 * kThreads));
    k_cp_async<<<1, kThreads>>>(d_in, d_out, mode);
    WG_CHECK(cudaGetLastError());
    WG_CHECK(cudaDeviceSynchronize());
    std::vector<uint4> o(2 * kThreads);
    WG_CHECK(cudaMemcpy(o.data(), d_out, sizeof(uint4) * 2 * kThreads, cudaMemcpyDeviceToHost));
    Checker c(names[mode]);
    for (int t = 0; t < kThreads; ++t) {
      const uint4 a = h[(t + 5) % kThreads];
      const uint4 b = h[kThreads + (t * 3) % kThreads];
      c.eq(8 * t + 0, o[t].x, a.x);
      c.eq(8 * t + 1, o[t].y, a.y);
      c.eq(8 * t + 2, o[t].z, a.z);
      c.eq(8 * t + 3, o[t].w, a.w);
      c.eq(8 * t + 4, o[kThreads + t].x, b.x);
      c.eq(8 * t + 5, o[kThreads + t].y, b.y);
      c.eq(8 * t + 6, o[kThreads + t].z, b.z);
      c.eq(8 * t + 7, o[kThreads + t].w, b.w);
    }
    wg_report(c);
  }
  WG_CHECK(cudaFree(d_in));
  WG_CHECK(cudaFree(d_out));
  return wg_finish();
}
