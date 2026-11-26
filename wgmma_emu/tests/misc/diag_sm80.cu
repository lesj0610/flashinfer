// Diagnostic (not a pass/fail test): what native sm_80 hardware returns for .relu on negative
// and negative-zero results, which the e2e expectations accept either way ([inference] in
// e2e_cvt.cu / e2e_arith.cu: PTX says "clamps the result to 0 if negative" without the sign).
// Fork-only. Owner: T2. Plain sm_80 code, not translated:
//   nvcc -arch=sm_80 -std=c++17 -O3 -I include tests/misc/diag_sm80.cu -o diag_sm80
// Output: one "WGEMU-DIAG <instruction> in=<inputs> out=<bits>" line per case.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const float kIn[] = {-1.5f, -0.0f, -1e-30f, 0.0f, 2.0f, __builtin_nanf("")};
constexpr int kN = sizeof(kIn) / sizeof(kIn[0]);

// out[4 i + k]: k = 0 cvt.rn.relu.f16x2.f32 (x, x), 1 cvt.rn.relu.bf16x2.f32 (x, x),
// 2 fma.rn.relu.f16 (x * 1 + -0), 3 fma.rn.relu.bf16 (x * 1 + -0).
__global__ void k_diag(const float* in, uint32_t* out) {
  const int i = threadIdx.x;
  if (i >= kN) return;
  const float x = in[i];
  uint32_t r0, r1;
  uint16_t h, hb;
  asm("cvt.rn.relu.f16x2.f32 %0, %1, %1;" : "=r"(r0) : "f"(x));
  asm("cvt.rn.relu.bf16x2.f32 %0, %1, %1;" : "=r"(r1) : "f"(x));
  asm("{ .reg .b16 a, one, nz; cvt.rn.f16.f32 a, %1; mov.b16 one, 0x3C00; mov.b16 nz, 0x8000;\n\t"
      "fma.rn.relu.f16 %0, a, one, nz; }"
      : "=h"(h)
      : "f"(x));
  asm("{ .reg .b16 a, one, nz; cvt.rn.bf16.f32 a, %1; mov.b16 one, 0x3F80; mov.b16 nz, 0x8000;\n\t"
      "fma.rn.relu.bf16 %0, a, one, nz; }"
      : "=h"(hb)
      : "f"(x));
  out[4 * i] = r0;
  out[4 * i + 1] = r1;
  out[4 * i + 2] = h;
  out[4 * i + 3] = hb;
}

int main() {
  float* din;
  uint32_t* dout;
  if (cudaMalloc(&din, sizeof(kIn)) != cudaSuccess ||
      cudaMalloc(&dout, 4 * kN * sizeof(uint32_t)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed\n");
    return 2;
  }
  cudaMemcpy(din, kIn, sizeof(kIn), cudaMemcpyHostToDevice);
  k_diag<<<1, 32>>>(din, dout);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(e));
    return 2;
  }
  uint32_t out[4 * kN];
  cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  static const char* kName[4] = {"cvt.rn.relu.f16x2.f32", "cvt.rn.relu.bf16x2.f32",
                                 "fma.rn.relu.f16(x,1,-0)", "fma.rn.relu.bf16(x,1,-0)"};
  for (int i = 0; i < kN; ++i) {
    uint32_t bits;
    memcpy(&bits, &kIn[i], 4);
    for (int k = 0; k < 4; ++k)
      printf("WGEMU-DIAG %s in=%08x out=%08x\n", kName[k], bits, out[4 * i + k]);
  }
  cudaFree(din);
  cudaFree(dout);
  return 0;
}
