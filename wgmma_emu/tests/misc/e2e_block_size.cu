// End-to-end check of a blocks-as-clusters kernel (.blocksareclusters with .reqntid and
// .reqnctapercluster, ptx.txt 36944-36962), emitted for CUDA's __block_size__ attribute.
// The core (T1) strips the directive for a cluster of one CTA. Fork-only. Owner: T2.
// Built for sm_90a by tools/nvcc_emu.
//
// [inference] __block_size__((32, 1, 1), (1, 1, 1)) emits .reqntid 32, 1, 1,
// .reqnctapercluster 1, 1, 1 and .blocksareclusters; the manifest covers check it on the
// original PTX. The launch goes through cudaLaunchKernel with the block size given explicitly;
// with one CTA per cluster the number of clusters equals the number of blocks. Needs a CUDA
// front end that knows __block_size__ and Ctx.cluster_info (T1); kept in its own program.
#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kOut = 4;

__global__ void __block_size__((32, 1, 1), (1, 1, 1)) k_bac(uint32_t* out) {
  uint32_t* o = out + kOut * (blockIdx.x * 32 + threadIdx.x);
  asm volatile("{ .reg .pred p; mov.pred p, %%is_explicit_cluster; selp.u32 %0, 1, 0, p; }"
               : "=r"(o[0]));
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(o[1]));
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(o[2]));
  o[3] = blockDim.x;
}

int main() {
  const int clusters = 5;
  Dev<uint32_t> out((size_t)kOut * clusters * 32);
  uint32_t* p = out.p;
  void* args[] = {&p};
  MISC_CUDA(cudaLaunchKernel((const void*)k_bac, dim3(clusters), dim3(32), args, 0, 0));
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it("misc.e2e.block_size.blocksareclusters");
  for (int b = 0; b < clusters; ++b)
    for (int t = 0; t < 32; ++t) {
      const uint32_t* v = &o[kOut * (b * 32 + t)];
      const uint32_t want[kOut] = {1, (uint32_t)b, (uint32_t)clusters, 32};
      for (int k = 0; k < kOut; ++k)
        it.check(v[k] == want[k], "cluster=%d thread=%d field=%d got=%u want=%u", b, t, k, v[k],
                 want[k]);
    }
  return finish();
}
