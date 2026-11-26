// Negative check-mode test: cp.async.bulk.shared::cluster.shared::cta copies into the shared
// memory of another CTA of the cluster; with the emulated cluster size of 1 there is no other CTA
// and the copy is undefined. Check mode must report WGEMU-VIOLATION
// cp.async.bulk.shared::cluster.shared::cta-needs-remote-cta (the other modes stop the kernel).
#include "async_common.cuh"

__global__ void k_s2s(uint32_t* out) {
  __shared__ alignas(128) uint32_t src[32], dst[32];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t t = threadIdx.x;
  const uint32_t b = smem_u32(&bar);
  src[t] = 0x7000u + t;
  dst[t] = 0;
  if (t == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (t == 0) {
    mbar_arrive_expect_tx(b, 128);
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
        " [%0], [%1], 128, [%2];" ::"r"(smem_u32(dst)),
        "r"(smem_u32(src)), "r"(b)
        : "memory");
  }
  mbar_wait(b, 0);
  out[t] = dst[t];
}

int main() {
  uint32_t* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 128));
  k_s2s<<<1, 32>>>(d);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> h(32);
  WG_CHECK(cudaMemcpy(h.data(), d, 128, cudaMemcpyDeviceToHost));
  Checker ck("async.neg.bulk_shared_to_cluster_without_remote_cta");
  for (int i = 0; i < 32; ++i) ck.eq(i, h[i], 0x7000u + i);
  wg_report(ck);
  WG_CHECK(cudaFree(d));
  return wg_finish();
}
