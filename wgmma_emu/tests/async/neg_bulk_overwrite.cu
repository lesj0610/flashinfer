// Negative check-mode test: a bulk shared->global copy whose shared-memory source is overwritten
// before cp.async.bulk.wait_group.read retires it. The emulated copy is synchronous, so the
// result looks right; check mode must report
// WGEMU-VIOLATION bulk-source-overwritten-before-wait_group.read (ptx.txt 16801-16830: the source
// may be modified only after wait_group.read).
#include "async_common.cuh"

__global__ void k_overwrite(uint32_t* g) {
  __shared__ alignas(128) uint32_t s[64];
  const uint32_t t = threadIdx.x;
  if (t == 0) {
    for (int i = 0; i < 64; ++i) s[i] = 0x5000u + i;
    fence_proxy_async_smem();
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" ::"l"(g),
                 "r"(smem_u32(s))
                 : "memory");
    bulk_commit();
    s[5] = 0xBADu;  // program error: the copy may still be reading s
    bulk_wait_read0();
  }
}

int main() {
  uint32_t* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 256));
  k_overwrite<<<1, 32>>>(d);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> h(64);
  WG_CHECK(cudaMemcpy(h.data(), d, 256, cudaMemcpyDeviceToHost));
  Checker ck("async.neg.bulk_source_overwrite");
  for (int i = 0; i < 64; ++i) ck.eq(i, h[i], 0x5000u + i);
  wg_report(ck);
  WG_CHECK(cudaFree(d));
  return wg_finish();
}
