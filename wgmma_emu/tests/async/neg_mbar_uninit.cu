// Negative check-mode test: an arrive on an mbarrier object that was never initialized
// (ptx.txt 20873: operations other than mbarrier.init on an uninitialized object are
// undefined). Check mode must report WGEMU-VIOLATION mbarrier-uninitialized; the zeroed state
// has pending count 0, so the same arrive also reports mbarrier-pending-underflow.
#include "async_common.cuh"

__global__ void k_uninit(uint64_t* out) {
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x == 0) bar = 0;  // zeroed, not initialized
  __syncthreads();
  if (threadIdx.x == 0) *out = mbar_arrive(smem_u32(&bar));
}

int main() {
  uint64_t* d = nullptr;
  WG_CHECK(cudaMalloc(&d, 8));
  k_uninit<<<1, 32>>>(d);
  WG_CHECK(cudaGetLastError());
  WG_CHECK(cudaDeviceSynchronize());
  Checker ck("async.neg.mbarrier_uninitialized");
  ck.ok(0, true);
  wg_report(ck);
  WG_CHECK(cudaFree(d));
  return wg_finish();
}
