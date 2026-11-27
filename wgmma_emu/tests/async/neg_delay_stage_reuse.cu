// Negative E1 (WGEMU_ASYNC_DELAY=1) tests of a one-stage full / empty mbarrier pipeline. Thread 0
// is the producer (bulk global->shared copies completing on `full`), all 128 threads consume and
// arrive on `empty` (count 128) when they are done with the stage. __syncthreads fixes the order,
// so every run takes the same path.
//
// REUSE=1: the producer refills the stage without waiting on `empty`, while the consumers still
//          read the previous round. The refill copies the same tile, so the exact mode (copy at
//          issue) hides the error; the delay mode poisons the stage at issue, so the reads after
//          the refill see 0xFFFFFFFF: result mismatch.
// REUSE=2: the producer issues round 1 into the stage before any thread observed round 0 on
//          `full`: two pending copies own the same words -> WGEMU-VIOLATION async-delay-overlap.
//          The consumers wait for round 1 and read its data, so the result is right.
// REUSE=3: the consumers arrive on `empty` before they finish reading (early release); the
//          producer waits on `empty` correctly and refills the stage with the same tile; the
//          late reads see the poison in delay mode: result mismatch.
#include "async_common.cuh"

#ifndef REUSE
#define REUSE 1
#endif

constexpr int kThreads = 128;
constexpr int kWords = 64;  // 256-byte stage

__device__ __forceinline__ void load_stage(uint32_t dst, const uint32_t* src, uint32_t bar) {
  mbar_arrive_expect_tx(bar, kWords * 4);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(dst),
      "l"(src), "r"(kWords * 4), "r"(bar)
      : "memory");
}

__global__ void k_reuse(const uint32_t* g, uint32_t* out) {
  __shared__ alignas(128) uint32_t stage[kWords];
  __shared__ alignas(8) uint64_t bars[2];
  const uint32_t t = threadIdx.x;
  const uint32_t full = smem_u32(&bars[0]);
  const uint32_t empty = smem_u32(&bars[1]);
  const uint32_t dst = smem_u32(stage);
  if (t == 0) {
    mbar_init(full, 1);
    mbar_init(empty, kThreads);
    fence_mbar_init();
  }
  __syncthreads();
#if REUSE == 2
  if (t == 0) {
    load_stage(dst, g, full);  // round 0
    // program error: no wait on `empty` (nobody has even observed round 0)
    load_stage(dst, g + kWords, full);  // round 1
  }
  __syncthreads();
  mbar_wait(full, 1);
  out[t] = stage[t % kWords];
  out[kThreads + t] = stage[(t * 7u) % kWords];
#else
  if (t == 0) load_stage(dst, g, full);  // round 0
  mbar_wait(full, 0);
  out[t] = stage[t % kWords];  // first part of the round-0 work
#if REUSE == 3
  mbar_arrive(empty);  // program error: released before the last read of the stage
#endif
  __syncthreads();
  if (t == 0) {
#if REUSE == 3
    mbar_wait(empty, 0);
#endif
    // REUSE == 1, program error: no wait on `empty` before the refill
    load_stage(dst, g, full);  // round 1: the same tile again
  }
  __syncthreads();
  out[kThreads + t] = stage[(t * 7u) % kWords];  // last read of the round-0 data
#if REUSE == 1
  mbar_arrive(empty);
#endif
  mbar_wait(full, 1);
#endif
}

int main() {
  std::vector<uint32_t> h(2 * kWords);
  for (int i = 0; i < 2 * kWords; ++i) h[i] = wg_hash32(i + 101u) & 0x7FFFFFFFu;
  uint32_t *d_in, *d_out;
  WG_CHECK(cudaMalloc(&d_in, 4 * 2 * kWords));
  WG_CHECK(cudaMalloc(&d_out, 4 * 2 * kThreads));
  WG_CHECK(cudaMemcpy(d_in, h.data(), 4 * 2 * kWords, cudaMemcpyHostToDevice));
  WG_CHECK(cudaMemset(d_out, 0, 4 * 2 * kThreads));
  k_reuse<<<1, kThreads>>>(d_in, d_out);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(2 * kThreads);
  WG_CHECK(cudaMemcpy(o.data(), d_out, 4 * 2 * kThreads, cudaMemcpyDeviceToHost));
  const uint32_t* want = h.data() + (REUSE == 2 ? kWords : 0);  // REUSE=2 reads round 1
  Checker first("async.delay.stage_reuse.first_read");
  for (int t = 0; t < kThreads; ++t) first.eq(t, o[t], want[t % kWords]);
  wg_report(first);
  Checker last("async.delay.stage_reuse.last_read");
  for (int t = 0; t < kThreads; ++t) last.eq(t, o[kThreads + t], want[(t * 7) % kWords]);
  wg_report(last);
  WG_CHECK(cudaFree(d_in));
  WG_CHECK(cudaFree(d_out));
  return wg_finish();
}
