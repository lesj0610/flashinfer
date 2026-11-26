// Sample SM90 program used to produce the translator test fixture (PTX only).
#include <cstdint>
#include <cstdio>

__device__ __noinline__ uint32_t elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %1;\n"
      "@%%px mov.s32 %0, 1;\n"
      "}\n"
      : "+r"(pred)
      : "r"(0xffffffffu));
  return pred;
}

__device__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred                P1; \n"
      "LAB_WAIT: \n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; \n"
      "@P1                       bra DONE; \n"
      "bra     LAB_WAIT; \n"
      "DONE: \n"
      "}"
      :
      : "r"(bar), "r"(phase));
}

__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(128, 1)
    sample_kernel(float* out, const __grid_constant__ uint64_t tmap_dummy, uint64_t desc_a,
                  uint64_t desc_b, int n) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (elect_one()) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(&tmap_dummy),
          "r"(bar_addr), "r"(0), "r"(n)
        : "memory");
  }
  mbar_wait(bar_addr, 0);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int it = 0; it < n; ++it) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(it));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  float acc = d[0] + d[1] + d[2] + d[3] + static_cast<float>(rank);
  out[threadIdx.x] = acc;
  if (threadIdx.x == 0) printf("WGEMU-SAMPLE %f\n", acc);
}
