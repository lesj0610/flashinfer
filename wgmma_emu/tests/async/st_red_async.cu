// st.async / red.async (ptx.txt 13589 / 13642-13645, 20451 / 20530-20533). Both write into the
// shared memory of another CTA of the cluster: st.async needs %cluster_nctarank > 1 and red.async a
// target CTA other than the executing one, so with the emulated cluster size of 1 every execution
// is undefined.
//
// Default build (positive test): the instructions sit behind a %cluster_nctarank > 1 guard that
// never holds, so the translation is covered and the guard value is checked; nothing executes.
// -DWGEMU_TEST_FORCE_REMOTE=1 (negative check-mode test): the guard is bypassed, every
// instruction executes against the own CTA and the runtime must report WGEMU-VIOLATION
// st.async-needs-cluster-nctarank-gt-1 / red.async-needs-remote-cta.
#include "async_common.cuh"

#ifndef WGEMU_TEST_FORCE_REMOTE
#define WGEMU_TEST_FORCE_REMOTE 0
#endif

__global__ void k_remote(uint32_t* out, int force) {
  __shared__ alignas(16) uint32_t buf[16];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  uint32_t n;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n));
  if (t < 16) buf[t] = 0x11110000u + t;
  const uint32_t b = smem_u32(&bar);
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  __syncthreads();
  if (t == 0 && (n > 1 || force)) {
    // Own-CTA shared::cta addresses are valid shared::cluster addresses of this CTA.
    const uint32_t a = smem_u32(buf);
    mbar_arrive_expect_tx(b, 4 + 8 + 16 + 8 + 4 + 4 + 8);
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];" ::"r"(a),
        "r"(0xA1u), "r"(b)
        : "memory");
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.u32 [%0], {%1, %2}, [%3];" ::"r"(
            a + 8),
        "r"(0xA2u), "r"(0xA3u), "r"(b)
        : "memory");
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4},"
        " [%5];" ::"r"(a + 16),
        "r"(0xA4u), "r"(0xA5u), "r"(0xA6u), "r"(0xA7u), "r"(b)
        : "memory");
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];" ::"r"(a + 32),
        "l"(0xA8A9ull), "r"(b)
        : "memory");
    asm volatile(
        "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32"
        " [%0], %1, [%2];" ::"r"(a + 40),
        "r"(5u), "r"(b)
        : "memory");
    asm volatile(
        "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32"
        " [%0], %1, [%2];" ::"r"(a + 44),
        "r"(-3), "r"(b)
        : "memory");
    asm volatile(
        "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64"
        " [%0], %1, [%2];" ::"r"(a + 48),
        "l"(7ull), "r"(b)
        : "memory");
    mbar_wait(b, 0);
  }
  __syncthreads();
  if (t == 0) out[16] = n;
  if (t < 16) out[t] = buf[t];
}

int main() {
  uint32_t* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 17 * 4));
  WG_CHECK(cudaMemset(d, 0, 17 * 4));
  k_remote<<<1, 32>>>(d, WGEMU_TEST_FORCE_REMOTE);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> h(17);
  WG_CHECK(cudaMemcpy(h.data(), d, 17 * 4, cudaMemcpyDeviceToHost));
  Checker ck("async.st_red_async.guarded");
  ck.eq(16, h[16], 1u);  // %cluster_nctarank of the emulated 1x1x1 cluster
  std::vector<uint32_t> want(16);
  for (uint32_t i = 0; i < 16; ++i) want[i] = 0x11110000u + i;
  if (WGEMU_TEST_FORCE_REMOTE) {  // every op ran against the own CTA
    want[0] = 0xA1u;
    want[2] = 0xA2u;
    want[3] = 0xA3u;
    for (uint32_t i = 0; i < 4; ++i) want[4 + i] = 0xA4u + i;
    want[8] = 0xA8A9u;
    want[9] = 0u;
    want[10] += 5u;
    want[11] = (uint32_t)-3;  // min.s32 against a positive old value
    want[12] += 7u;           // add.u64 on (buf[13]:buf[12]), no carry
  }
  for (uint32_t i = 0; i < 16; ++i) ck.eq(i, h[i], want[i]);
  wg_report(ck);
  WG_CHECK(cudaFree(d));
  return wg_finish();
}
