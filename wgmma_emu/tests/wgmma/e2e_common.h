// Shared code of the generated wgmma end-to-end tests (tests/wgmma/e2e/*.cu, written by
// tests/wgmma/gen_e2e.py) and of tests/wgmma/cute_gmma.cu. Owner: W.
//
// The tests are compiled for sm_90a by tools/nvcc_emu (the build harness
// tests/e2e/common/build_and_run.py), i.e. their wgmma PTX goes through the translator and runs
// on SM80 through runtime/lib_wgmma.cu. Nothing here calls the emulation library directly.
//
// Kernel contract (generated kernels):
//   * one warpgroup (128 threads), dynamic shared memory e2e::kSmemBytes;
//   * e2e::stage() copies the operand image to a 1024-byte aligned region and returns the
//     absolute descriptors (region-relative descriptor + region address >> 4);
//   * per thread, input words a0..a3, sp-meta, then the C registers; output the D registers.
// Host side: e2e::Harness lays out a wgref::Problem (ref_model.h), launches, and compares with
// ref_exact (exact and check modes) or ref_fast (fast mode). The mode the binary was built
// for is not visible to it (the harness only sets WGEMU_MODE for nvcc_emu), so every binary
// first runs a probe kernel on an input whose exact and fast results differ and takes the
// mode from the probe result; WGEMU_MODE, when set at run time, must agree.
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ref_model.h"

namespace e2e {

using wgemu::EType;

constexpr uint32_t kRegionBytes = 163840;
constexpr size_t kSmemBytes = 1024 + kRegionBytes;
constexpr int kInWords = 5;  // a0..a3, sp-meta (then the C registers)

struct KArgs {
  const uint8_t* img;     // operand image, region relative, multiple of 16 bytes
  const uint32_t* rin;    // per thread: kInWords + C registers
  uint32_t* rout;         // per thread: output registers
  uint64_t adesc, bdesc;  // region-relative descriptors (adesc unused by RS kernels)
  uint32_t img_bytes;
  uint32_t scale_d;
  uint32_t flip;  // negative tests: region-relative byte of B(0, 0)
  uint32_t pad;
};

#if defined(__CUDACC__)
// Generic-proxy shared-memory writes must be made visible to the async proxy (wgmma operand
// reads) before the wgmma is issued; the tests are valid sm_90a programs, so they fence.
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Copies the image into the 1024-aligned dynamic shared-memory region (all 128 threads), returns
// the region; *adesc / *bdesc get the absolute descriptors.
__device__ __forceinline__ uint8_t* stage(const KArgs& g, uint64_t* adesc, uint64_t* bdesc) {
  extern __shared__ __align__(16) uint8_t e2e_smem[];
  const uint32_t sh0 = static_cast<uint32_t>(__cvta_generic_to_shared(e2e_smem));
  const uint32_t pad = ((sh0 + 1023u) & ~1023u) - sh0;
  uint8_t* region = e2e_smem + pad;
  const uint4* src = reinterpret_cast<const uint4*>(g.img);
  uint4* dst = reinterpret_cast<uint4*>(region);
  for (uint32_t i = threadIdx.x; i < g.img_bytes / 16u; i += blockDim.x) dst[i] = src[i];
  fence_async_smem();
  __syncthreads();
  const uint64_t base4 = (sh0 + pad) >> 4;
  if (((g.bdesc & 0x3FFFu) + base4) > 0x3FFFu || ((g.adesc & 0x3FFFu) + base4) > 0x3FFFu) __trap();
  *adesc = g.adesc + base4;
  *bdesc = g.bdesc + base4;
  return region;
}
#endif

using KernelFn = void (*)(KArgs);

// One generated kernel: a single wgmma.mma_async of this form.
struct Kern {
  KernelFn fn;
  const char* tag;
  int n;
  EType at, bt, dt;
  bool sparse, rs, trans_a, trans_b, neg_a, neg_b, satfinite;
  int sp_sel;
};

// One launch of a kernel: run-time layout of A and B, scale-d, base_offset, input seed.
struct Run {
  int kernel;
  int sw_a, sw_b, order_a, order_b, kofs_a, kofs_b, gap_a, gap_b;
  int scale_d, base_off;
  unsigned long long seed;
};

#define E2E_CK(x)                                                                           \
  do {                                                                                      \
    cudaError_t e_ = (x);                                                                   \
    if (e_ != cudaSuccess) {                                                                \
      printf("WGEMU-ERROR cuda %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      fflush(stdout);                                                                       \
      exit(2);                                                                              \
    }                                                                                       \
  } while (0)

inline wgref::Op op_of(const Kern& k) {
  wgref::Op o;
  o.n = k.n;
  o.at = k.at;
  o.bt = k.bt;
  o.dt = k.dt;
  o.sparse = k.sparse;
  o.neg_a = k.neg_a;
  o.neg_b = k.neg_b;
  o.satfinite = k.satfinite;
  o.sp_sel = k.sp_sel;
  return o;
}

class Harness {
 public:
  explicit Harness(const char* prefix) : prefix_(prefix) {}

  // Launches fn once for P with the given layouts; out gets nout words per thread. Returns
  // false (with *why) when the operand image cannot be built.
  bool launch(KernelFn fn, const wgref::Problem& P, bool rs, const wgref::LayoutSel& la,
              const wgref::LayoutSel& lb, int base_off, uint64_t seed, int nout,
              std::vector<uint32_t>* out, std::string* why) {
    if (!attr_done_.count(fn)) {
      E2E_CK(
          cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmemBytes)));
      attr_done_.insert(fn);
    }
    std::vector<uint8_t> img;
    wgref::Placement pl;
    if (!wgref::place_problem(P, rs, la, lb, base_off, kRegionBytes, seed, img, &pl, nullptr, why))
      return false;
    const wgref::Op& o = P.op;
    const int nr = wgref::op_nregs(o);
    std::vector<uint32_t> rin;
    wgref::Rng r(seed ^ 0xC0FFEEull);
    std::vector<uint32_t> garbage(size_t(64) * o.n);
    for (auto& x : garbage) x = r.u32();
    const std::vector<uint32_t>& cm = o.scale_d ? P.C : garbage;
    for (int t = 0; t < 128; ++t) {
      uint32_t a[4] = {r.u32(), r.u32(), r.u32(), r.u32()};
      if (rs) wgref::pack_a(P, t, a);
      const uint32_t meta = o.sparse ? wgref::pack_meta(P, t, r.u32()) : r.u32();
      uint32_t regs[128];
      wgref::pack_c(o, cm, t, regs);
      rin.insert(rin.end(), a, a + 4);
      rin.push_back(meta);
      rin.insert(rin.end(), regs, regs + nr);
    }
    KArgs g = {};
    uint8_t* d_img;
    uint32_t *d_in, *d_out;
    E2E_CK(cudaMalloc(&d_img, img.size()));
    E2E_CK(cudaMalloc(&d_in, rin.size() * 4));
    E2E_CK(cudaMalloc(&d_out, size_t(128) * nout * 4));
    E2E_CK(cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice));
    E2E_CK(cudaMemcpy(d_in, rin.data(), rin.size() * 4, cudaMemcpyHostToDevice));
    E2E_CK(cudaMemset(d_out, 0xCD, size_t(128) * nout * 4));
    g.img = d_img;
    g.rin = d_in;
    g.rout = d_out;
    g.adesc = rs ? 0ull : pl.adesc;
    g.bdesc = pl.bdesc;
    g.img_bytes = pl.bytes;
    g.scale_d = o.scale_d ? 1u : 0u;
    g.flip = pl.b_first_addr;
    fn<<<1, 128, kSmemBytes>>>(g);
    E2E_CK(cudaGetLastError());
    E2E_CK(cudaDeviceSynchronize());
    out->assign(size_t(128) * nout, 0u);
    E2E_CK(cudaMemcpy(out->data(), d_out, out->size() * 4, cudaMemcpyDeviceToHost));
    E2E_CK(cudaFree(d_img));
    E2E_CK(cudaFree(d_in));
    E2E_CK(cudaFree(d_out));
    return true;
  }

  // D (64 x n) of one launch of a single-accumulator kernel.
  bool run_d(KernelFn fn, const wgref::Problem& P, bool rs, const wgref::LayoutSel& la,
             const wgref::LayoutSel& lb, int base_off, uint64_t seed, std::vector<uint32_t>* D,
             std::string* why) {
    const int nr = wgref::op_nregs(P.op);
    std::vector<uint32_t> raw;
    if (!launch(fn, P, rs, la, lb, base_off, seed, nr, &raw, why)) return false;
    D->assign(size_t(64) * P.op.n, 0u);
    for (int t = 0; t < 128; ++t) wgref::unpack_d(P.op, t, raw.data() + size_t(t) * nr, *D);
    return true;
  }

  // Mode probe: k must be an f16 x f16 -> f32, N = 8, SS kernel without flags.
  void probe(const Kern& k) {
    const std::string name = prefix_ + ".mode_probe";
    wgref::Stat& st = stats_[name];
    ++st.cases;
    wgref::Op o = op_of(k);
    wgref::Problem P;
    std::vector<uint32_t> ex, fa;
    uint64_t seed = 0;
    for (uint64_t s = 1; s <= 4096 && seed == 0; ++s) {
      P = wgref::make_problem(o, 0x9E0B0000ull + s);
      ex = wgref::ref_exact(P);
      fa = wgref::ref_fast(P);
      if (ex != fa) seed = 0x9E0B0000ull + s;
    }
    if (seed == 0) {
      printf("WGEMU-MISMATCH name=%s no input separates exact and fast\n", name.c_str());
      ++st.mismatches;
      return;
    }
    wgref::LayoutSel la, lb;
    la.sw = lb.sw = 128;
    std::vector<uint32_t> got;
    std::string why;
    if (!run_d(k.fn, P, false, la, lb, 0, seed, &got, &why)) {
      printf("WGEMU-MISMATCH name=%s layout %s\n", name.c_str(), why.c_str());
      ++st.mismatches;
      return;
    }
    const bool is_exact = got == ex, is_fast = got == fa;
    const char* env = getenv("WGEMU_MODE");
    if (is_exact == is_fast) {
      printf("WGEMU-MISMATCH name=%s probe result matches %s reference\n", name.c_str(),
             is_exact ? "both" : "neither");
      ++st.mismatches;
      return;
    }
    fast_ = is_fast;
    probed_ = true;
    printf("WGEMU-INFO %s mode=%s\n", prefix_.c_str(), fast_ ? "fast" : "exact-or-check");
    if (env && *env) {
      const bool env_fast = strcmp(env, "fast") == 0;
      if (env_fast != fast_ ||
          (!env_fast && strcmp(env, "exact") != 0 && strcmp(env, "check") != 0)) {
        printf("WGEMU-MISMATCH name=%s WGEMU_MODE=%s but probe says %s\n", name.c_str(), env,
               fast_ ? "fast" : "exact/check");
        ++st.mismatches;
      }
    }
  }

  // One positive run: D compared bit for bit with the reference of the probed mode.
  void run(const Kern& k, const Run& r) {
    const std::string name = prefix_ + "." + k.tag;
    wgref::Stat& st = stats_[name];
    wgref::Op o = op_of(k);
    o.scale_d = r.scale_d != 0;
    const wgref::Problem P = wgref::make_problem(o, r.seed);
    wgref::LayoutSel la, lb;
    la.mn_major = k.trans_a;
    lb.mn_major = k.trans_b;
    la.sw = r.sw_a;
    lb.sw = r.sw_b;
    la.order = r.order_a;
    lb.order = r.order_b;
    la.kofs = r.kofs_a;
    lb.kofs = r.kofs_b;
    la.gap = r.gap_a;
    lb.gap = r.gap_b;
    char what[160];
    snprintf(what, sizeof(what), "swA=%d swB=%d oA=%d oB=%d sd=%d boff=%d seed=%llu", r.sw_a,
             r.sw_b, r.order_a, r.order_b, r.scale_d, r.base_off, r.seed);
    if (!probed_) {
      ++st.cases;
      ++st.mismatches;
      printf("WGEMU-MISMATCH name=%s %s mode unknown (probe failed)\n", name.c_str(), what);
      return;
    }
    std::vector<uint32_t> got;
    std::string why;
    if (!run_d(k.fn, P, k.rs, la, lb, r.base_off, r.seed, &got, &why)) {
      ++st.cases;
      ++st.mismatches;
      printf("WGEMU-MISMATCH name=%s %s layout-error %s\n", name.c_str(), what, why.c_str());
      return;
    }
    wgref::compare(name.c_str(), what, o, got, wgref::ref_mode(P, P.C, fast_), st);
  }

  wgref::Stat& stat(const std::string& item) { return stats_[prefix_ + "." + item]; }
  bool fast() const { return fast_; }
  bool probed() const { return probed_; }

  int finish() {
    bool ok = true;
    for (const auto& kv : stats_) {
      printf("WGEMU-TEST name=%s cases=%llu mismatches=%llu\n", kv.first.c_str(),
             (unsigned long long)kv.second.cases, (unsigned long long)kv.second.mismatches);
      if (kv.second.cases == 0 || kv.second.mismatches != 0) ok = false;
    }
    printf("WGEMU-TEST-END\n");
    fflush(stdout);
    return ok ? 0 : 1;
  }

 private:
  std::string prefix_;
  std::map<std::string, wgref::Stat> stats_;
  std::set<KernelFn> attr_done_;
  bool fast_ = false;
  bool probed_ = false;
};

}  // namespace e2e
