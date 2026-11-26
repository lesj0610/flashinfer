// Negative E1 (WGEMU_ASYNC_DELAY=1) test: shared memory read without waiting on the mbarrier of a
// bulk global->shared copy. Thread 0 issues the copy; after __syncthreads every thread reads the
// destination before any thread observed the barrier phase (ptx.txt: the data is visible only
// after a wait on the mbarrier observes the phase completion). The exact mode copies at issue,
// so the program error gives the right data there (manifest entry *.exact_baseline); the delay
// mode leaves WGEMU_DL_POISON in the destination until a wait observes the phase, so the early
// reads see 0xFFFFFFFF and the result mismatches. The late wait then delivers the data.
#include "async_common.cuh"

constexpr int kThreads = 128;
constexpr int kWords = 128;  // 512-byte copy

__global__ void k_nowait(const uint32_t* g, uint32_t* out) {
  __shared__ alignas(128) uint32_t s[kWords];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b, kWords * 4);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
            "r"(smem_u32(s)),
        "l"(g), "r"(kWords * 4), "r"(b)
        : "memory");
  }
  __syncthreads();
  out[t] = s[(t * 5u) % kWords];  // program error: no mbarrier wait before the read
  mbar_wait(b, 0);
  out[kThreads + t] = s[(t * 3u) % kWords];  // after the wait: always the copied data
}

int main() {
  std::vector<uint32_t> h(kWords);
  for (int i = 0; i < kWords; ++i) h[i] = wg_hash32(i + 11u) & 0x7FFFFFFFu;  // never the poison
  uint32_t *d_in, *d_out;
  WG_CHECK(cudaMalloc(&d_in, 4 * kWords));
  WG_CHECK(cudaMalloc(&d_out, 4 * 2 * kThreads));
  WG_CHECK(cudaMemcpy(d_in, h.data(), 4 * kWords, cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_out, 0, 4 * 2 * kThreads));
  k_nowait<<<1, kThreads>>>(d_in, d_out);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(2 * kThreads);
  WG_CHECK(cudaMemcpy(o.data(), d_out, 4 * 2 * kThreads, cudaMemcpyDeviceToHost));
  Checker before("async.delay.nowait.read_before_wait");
  for (int t = 0; t < kThreads; ++t) before.eq(t, o[t], h[(t * 5) % kWords]);
  wg_report(before);
  Checker after("async.delay.nowait.read_after_wait");
  for (int t = 0; t < kThreads; ++t) after.eq(t, o[kThreads + t], h[(t * 3) % kWords]);
  wg_report(after);
  WG_CHECK(cudaFree(d_in));
  WG_CHECK(cudaFree(d_out));
  return wg_finish();
}
