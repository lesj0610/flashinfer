// CuTe SM90 GMMA atoms end to end (manifest entry wgmma.cute.gmma, written by
// tests/wgmma/gen_e2e.py). Owner: W.
//
// Built by tests/e2e/common/build_and_run.py through tools/nvcc_emu (sm_90a, CUTLASS include
// path and --expt-relaxed-constexpr from the manifest), so the wgmma / wgmma.fence /
// commit_group / wait_group PTX that CuTe emits goes through the translator and runs on SM80.
//
// What this adds over tests/wgmma/test_direct.cu and the generated e2e programs: operand
// shared-memory layouts (GMMA::Layout_{K,MN}_{INTER,SW32,SW64,SW128}_Atom tiled with
// tile_to_shape), descriptors (make_gmma_desc through make_fragment_A/B, k-block descriptor
// offsets inside a swizzle atom) and register fragments (partition_A/B/C, RS A fragments) all
// come from CuTe, i.e. from the conventions real sm_90a code uses. Each case is one warpgroup
// running the CUTLASS mainloop pattern over KT / K k-blocks:
//   warpgroup_fence_operand(C); warpgroup_arrive();
//   for k-block: cute::gemm(tiled_mma, A(_,_,k), B(_,_,k), C); accumulate_ = One;
//   warpgroup_commit_batch(); warpgroup_wait<0>(); warpgroup_fence_operand(C);
// with accumulate_ = Zero or One for the first k-block (both are run). The host reference is
// the chain of per-k-block wgmma references of tests/wgmma/ref_model.h (exact for the exact and
// check builds, the fast decomposition for the fast build; the build mode is found by the
// e2e::Harness mode probe, which here issues a CuTe SM90_64x8x16_F32F16F16_SS atom).
//
// Not covered here (covered by test_direct.cu and the generated e2e programs): sparse wgmma
// (CuTe sparse atoms use their own SparseElem / metadata layouts), b1, u8, e5m2 and the
// mixed-fp8 / mixed-int8 type pairs, satfinite only on s8 x s8.
//
// Output: WGEMU-INFO wgmma.cute mode=..., then WGEMU-TEST name=wgmma.cute.<case> cases=...
// mismatches=... per case and WGEMU-TEST-END; exit 0 iff every case ran with no mismatch.
#include <cute/tensor.hpp>

#include "e2e_common.h"

namespace {

using wgemu::EType;
namespace G = cute::SM90::GMMA;
using cute::bfloat16_t;
using cute::float_e4m3_t;
using cute::half_t;
using cute::tfloat32_t;

constexpr G::Major kK = G::Major::K;
constexpr G::Major kMN = G::Major::MN;
constexpr G::ScaleIn kNeg = G::ScaleIn::Neg;
constexpr G::ScaleIn kOne = G::ScaleIn::One;

template <class T>
using KInter = G::Layout_K_INTER_Atom<T>;
template <class T>
using KSw32 = G::Layout_K_SW32_Atom<T>;
template <class T>
using KSw64 = G::Layout_K_SW64_Atom<T>;
template <class T>
using KSw128 = G::Layout_K_SW128_Atom<T>;
template <class T>
using MNInter = G::Layout_MN_INTER_Atom<T>;
template <class T>
using MNSw32 = G::Layout_MN_SW32_Atom<T>;
template <class T>
using MNSw64 = G::Layout_MN_SW64_Atom<T>;
template <class T>
using MNSw128 = G::Layout_MN_SW128_Atom<T>;

template <int Bits>
struct RawOf;
template <>
struct RawOf<8> {
  using type = uint8_t;
};
template <>
struct RawOf<16> {
  using type = uint16_t;
};
template <>
struct RawOf<32> {
  using type = uint32_t;
};

// One CuTe case: atom, CuTe value types, smem layout atoms of A (64 x KT) and B (N x KT), and
// the logical wgmma it issues (for the reference).
template <class Op_, class TA_, class TB_, class TC_, class AtomA_, class AtomB_, int N_, int KT_,
          bool RS_, EType AT_, EType BT_, EType DT_, bool NegA_ = false, bool NegB_ = false,
          bool Sat_ = false>
struct Cfg {
  using Op = Op_;
  using TA = TA_;
  using TB = TB_;
  using TC = TC_;
  using AtomA = AtomA_;
  using AtomB = AtomB_;
  using UA = typename RawOf<cute::sizeof_bits<TA>::value>::type;
  using UB = typename RawOf<cute::sizeof_bits<TB>::value>::type;
  using UC = typename RawOf<cute::sizeof_bits<TC>::value>::type;
  static constexpr int N = N_, KT = KT_;
  static constexpr bool RS = RS_;
  static constexpr EType AT = AT_, BT = BT_, DT = DT_;
  static constexpr bool NegA = NegA_, NegB = NegB_, Sat = Sat_;
  static constexpr uint32_t kABytes =
      (64u * KT * cute::sizeof_bits<TA>::value / 8u + 1023u) / 1024u * 1024u;
  static constexpr uint32_t kBBytes = uint32_t(N) * KT * cute::sizeof_bits<TB>::value / 8u;
  static constexpr size_t kSmem = 1024 + kABytes + kBBytes;
};

// a_in: 64 x KT, b_in: N x KT, c_in / d_out: 64 x N, all row-major raw bits.
template <class C>
__global__ void __launch_bounds__(128, 1)
    cute_gmma_kernel(const typename C::UA* a_in, const typename C::UB* b_in,
                     const typename C::UC* c_in, typename C::UC* d_out, int first_zero) {
  using namespace cute;
  using TA = typename C::TA;
  using TB = typename C::TB;
  using TC = typename C::TC;
  extern __shared__ __align__(16) uint8_t cute_smem[];
  const uint32_t sh0 = static_cast<uint32_t>(__cvta_generic_to_shared(cute_smem));
  uint8_t* base = cute_smem + (((sh0 + 1023u) & ~1023u) - sh0);
  auto sA_layout = tile_to_shape(typename C::AtomA{}, make_shape(Int<64>{}, Int<C::KT>{}));
  auto sB_layout = tile_to_shape(typename C::AtomB{}, make_shape(Int<C::N>{}, Int<C::KT>{}));
  // tile_to_shape of a GMMA atom is compact: A takes 64 * KT elements (kABytes rounds it up to
  // the 1024-byte swizzle period so B starts on a period boundary).
  TA* pa = reinterpret_cast<TA*>(base);
  TB* pb = reinterpret_cast<TB*>(base + C::kABytes);
  Tensor sA = make_tensor(make_smem_ptr(pa), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(pb), sB_layout);
  // Same layouts over raw-bit views: the operands are written as bit patterns (no conversion).
  Tensor rA = make_tensor(make_smem_ptr(reinterpret_cast<typename C::UA*>(pa)), sA_layout);
  Tensor rB = make_tensor(make_smem_ptr(reinterpret_cast<typename C::UB*>(pb)), sB_layout);
  for (int i = threadIdx.x; i < 64 * C::KT; i += 128) rA(i / C::KT, i % C::KT) = a_in[i];
  for (int i = threadIdx.x; i < C::N * C::KT; i += 128) rB(i / C::KT, i % C::KT) = b_in[i];
  e2e::fence_async_smem();
  __syncthreads();

  auto tiled_mma = make_tiled_mma(typename C::Op{});
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor gC = make_tensor(make_gmem_ptr(reinterpret_cast<const TC*>(c_in)),
                          make_layout(make_shape(Int<64>{}, Int<C::N>{}), LayoutRight{}));
  Tensor gD = make_tensor(make_gmem_ptr(reinterpret_cast<TC*>(d_out)),
                          make_layout(make_shape(Int<64>{}, Int<C::N>{}), LayoutRight{}));
  Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
  Tensor tCgD = thr_mma.partition_C(gD);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // SS: descriptors, RS: registers
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // descriptors
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  constexpr int kKBlocks = decltype(size<2>(tCrA))::value;
  constexpr int kKAtom = decltype(size<2>(typename MMA_Traits<typename C::Op>::Shape_MNK{}))::value;
  static_assert(kKBlocks * kKAtom == C::KT, "k-blocks do not tile KT");
  // Element-wise moves keep every bit pattern (no vectorization assumptions on gmem).
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) tCrC(i) = tCgC(i);
  if constexpr (C::RS) {
    CUTE_UNROLL
    for (int i = 0; i < size(tCrA); ++i) tCrA(i) = tCsA(i);
  }

  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  tiled_mma.accumulate_ = first_zero ? G::ScaleOut::Zero : G::ScaleOut::One;
  CUTE_UNROLL
  for (int kb = 0; kb < kKBlocks; ++kb) {
    cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    tiled_mma.accumulate_ = G::ScaleOut::One;
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) tCgD(i) = tCrC(i);
}

// Mode probe kernel (e2e::KArgs contract): one CuTe SM90_64x8x16_F32F16F16_SS<K, K> atom.
using ProbeOp = cute::SM90_64x8x16_F32F16F16_SS<kK, kK>;
__global__ void __launch_bounds__(128, 1) cute_probe_kernel(e2e::KArgs g) {
  uint64_t adesc, bdesc;
  e2e::stage(g, &adesc, &bdesc);
  const uint32_t* in = g.rin + threadIdx.x * (e2e::kInWords + 4);
  float d[4];
  for (int i = 0; i < 4; ++i) d[i] = __uint_as_float(in[e2e::kInWords + i]);
  for (int i = 0; i < 4; ++i) cute::warpgroup_fence_operand(d[i]);
  cute::warpgroup_arrive();
  ProbeOp::fma(adesc, bdesc, d[0], d[1], d[2], d[3],
               g.scale_d ? G::ScaleOut::One : G::ScaleOut::Zero);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  for (int i = 0; i < 4; ++i) cute::warpgroup_fence_operand(d[i]);
  for (int i = 0; i < 4; ++i) g.rout[threadIdx.x * 4 + i] = __float_as_uint(d[i]);
}

template <class C>
void run_one(e2e::Harness& h, const char* tag, uint64_t seed, bool first_zero) {
  wgref::Stat& st = h.stat(tag);
  const std::string name = std::string("wgmma.cute.") + tag;
  char what[96];
  snprintf(what, sizeof(what), "first=%s seed=%llu", first_zero ? "zero" : "one",
           (unsigned long long)seed);
  wgref::Op op;
  op.n = C::N;
  op.at = C::AT;
  op.bt = C::BT;
  op.dt = C::DT;
  op.neg_a = C::NegA;
  op.neg_b = C::NegB;
  op.satfinite = C::Sat;
  const int K = wgref::op_k(op), KB = C::KT / K;
  if (!h.probed() || KB * K != C::KT || wgref::op_ka(op) != K || wgref::op_kb(op) != K) {
    ++st.cases;
    ++st.mismatches;
    printf("WGEMU-MISMATCH name=%s %s %s\n", name.c_str(), what,
           h.probed() ? "k-block shape does not match the reference op" : "mode unknown");
    return;
  }
  std::vector<wgref::Problem> P(KB);
  for (int j = 0; j < KB; ++j) {
    wgref::Op oj = op;
    oj.scale_d = j > 0 || !first_zero;
    P[j] = wgref::make_problem(oj, seed * 16u + uint64_t(j));
  }
  std::vector<typename C::UA> A(size_t(64) * C::KT);
  std::vector<typename C::UB> B(size_t(C::N) * C::KT);
  std::vector<typename C::UC> Cin(size_t(64) * C::N), Dout(size_t(64) * C::N);
  for (int j = 0; j < KB; ++j)
    for (int k = 0; k < K; ++k) {
      for (int m = 0; m < 64; ++m)
        A[size_t(m) * C::KT + j * K + k] = static_cast<typename C::UA>(P[j].A[size_t(m) * K + k]);
      for (int n = 0; n < C::N; ++n)
        B[size_t(n) * C::KT + j * K + k] = static_cast<typename C::UB>(P[j].B[size_t(n) * K + k]);
    }
  for (size_t i = 0; i < Cin.size(); ++i) Cin[i] = static_cast<typename C::UC>(P[0].C[i]);

  typename C::UA* dA;
  typename C::UB* dB;
  typename C::UC *dC, *dD;
  E2E_CK(cudaMalloc(&dA, A.size() * sizeof(A[0])));
  E2E_CK(cudaMalloc(&dB, B.size() * sizeof(B[0])));
  E2E_CK(cudaMalloc(&dC, Cin.size() * sizeof(Cin[0])));
  E2E_CK(cudaMalloc(&dD, Dout.size() * sizeof(Dout[0])));
  E2E_CK(cudaMemcpy(dA, A.data(), A.size() * sizeof(A[0]), cudaMemcpyHostToDevice));
  E2E_CK(cudaMemcpy(dB, B.data(), B.size() * sizeof(B[0]), cudaMemcpyHostToDevice));
  E2E_CK(cudaMemcpy(dC, Cin.data(), Cin.size() * sizeof(Cin[0]), cudaMemcpyHostToDevice));
  E2E_CK(cudaMemset(dD, 0xCD, Dout.size() * sizeof(Dout[0])));
  E2E_CK(cudaFuncSetAttribute(cute_gmma_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              int(C::kSmem)));
  cute_gmma_kernel<C><<<1, 128, C::kSmem>>>(dA, dB, dC, dD, first_zero ? 1 : 0);
  E2E_CK(cudaGetLastError());
  E2E_CK(cudaDeviceSynchronize());
  E2E_CK(cudaMemcpy(Dout.data(), dD, Dout.size() * sizeof(Dout[0]), cudaMemcpyDeviceToHost));
  E2E_CK(cudaFree(dA));
  E2E_CK(cudaFree(dB));
  E2E_CK(cudaFree(dC));
  E2E_CK(cudaFree(dD));

  std::vector<uint32_t> want = P[0].C;
  for (int j = 0; j < KB; ++j) want = wgref::ref_mode(P[j], want, h.fast());
  std::vector<uint32_t> got(Dout.begin(), Dout.end());
  wgref::compare(name.c_str(), what, op, got, want, st);
}

template <class C>
void run_cfg(e2e::Harness& h, const char* tag, uint64_t seed) {
  run_one<C>(h, tag, seed, false);
  run_one<C>(h, tag, seed + 1u, true);
}

// clang-format off
// f32 += f16 x f16, N = 64: every swizzle, both operands K-major, then both MN-major.
using F16KK128 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                     KSw128<half_t>, KSw128<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16KK64 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                    KSw64<half_t>, KSw64<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16KK32 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                    KSw32<half_t>, KSw32<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16KKI = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                   KInter<half_t>, KInter<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16MM128 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kMN, kMN>, half_t, half_t, float,
                     MNSw128<half_t>, MNSw128<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16MM64 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kMN, kMN>, half_t, half_t, float,
                    MNSw64<half_t>, MNSw64<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16MM32 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kMN, kMN>, half_t, half_t, float,
                    MNSw32<half_t>, MNSw32<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
using F16MMI = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kMN, kMN>, half_t, half_t, float,
                   MNInter<half_t>, MNInter<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32>;
// Smallest and largest N.
using F16N8 = Cfg<cute::SM90_64x8x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                  KInter<half_t>, KInter<half_t>, 8, 64, false, EType::F16, EType::F16, EType::F32>;
using F16N256 = Cfg<cute::SM90_64x256x16_F32F16F16_SS<kK, kK>, half_t, half_t, float,
                    KSw128<half_t>, KSw128<half_t>, 256, 64, false, EType::F16, EType::F16, EType::F32>;
// f16 += f16 x f16 with mixed majors.
using H16KM = Cfg<cute::SM90_64x64x16_F16F16F16_SS<kK, kMN>, half_t, half_t, half_t,
                  KSw64<half_t>, MNSw32<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F16>;
using H16MK = Cfg<cute::SM90_64x64x16_F16F16F16_SS<kMN, kK>, half_t, half_t, half_t,
                  MNSw128<half_t>, KInter<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F16>;
// bf16.
using BF16KK = Cfg<cute::SM90_64x64x16_F32BF16BF16_SS<kK, kK>, bfloat16_t, bfloat16_t, float,
                   KSw128<bfloat16_t>, KSw32<bfloat16_t>, 64, 64, false, EType::BF16, EType::BF16, EType::F32>;
using BF16MM = Cfg<cute::SM90_64x64x16_F32BF16BF16_SS<kMN, kMN>, bfloat16_t, bfloat16_t, float,
                   MNSw64<bfloat16_t>, MNInter<bfloat16_t>, 64, 64, false, EType::BF16, EType::BF16, EType::F32>;
// tf32 (K-major only).
using TF32A = Cfg<cute::SM90_64x64x8_F32TF32TF32_SS_TN<>, tfloat32_t, tfloat32_t, float,
                  KSw128<tfloat32_t>, KSw64<tfloat32_t>, 64, 32, false, EType::TF32, EType::TF32, EType::F32>;
using TF32B = Cfg<cute::SM90_64x64x8_F32TF32TF32_SS_TN<>, tfloat32_t, tfloat32_t, float,
                  KSw32<tfloat32_t>, KInter<tfloat32_t>, 64, 32, false, EType::TF32, EType::TF32, EType::F32>;
// e4m3 (K-major only).
using E4M3A = Cfg<cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>, float_e4m3_t, float_e4m3_t, float,
                  KSw128<float_e4m3_t>, KSw32<float_e4m3_t>, 64, 128, false, EType::E4M3, EType::E4M3, EType::F32>;
using E4M3B = Cfg<cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>, float_e4m3_t, float_e4m3_t, float,
                  KInter<float_e4m3_t>, KSw64<float_e4m3_t>, 64, 128, false, EType::E4M3, EType::E4M3, EType::F32>;
// s8 (K-major only), plain and saturating.
using S8A = Cfg<cute::SM90_64x64x32_S32S8S8_SS_TN, int8_t, int8_t, int32_t,
                KSw128<int8_t>, KSw64<int8_t>, 64, 128, false, EType::S8, EType::S8, EType::S32>;
using S8Sat = Cfg<cute::SM90_64x64x32_S32S8S8_SS_TN_SATURATE, int8_t, int8_t, int32_t,
                  KSw32<int8_t>, KInter<int8_t>, 64, 128, false, EType::S8, EType::S8, EType::S32,
                  false, false, true>;
// RS (A from registers, K-major in CuTe's register layout; B K- and MN-major).
using RSF32KK = Cfg<cute::SM90_64x64x16_F32F16F16_RS<kK, kK>, half_t, half_t, float,
                    KSw128<half_t>, KSw128<half_t>, 64, 64, true, EType::F16, EType::F16, EType::F32>;
using RSF32KM = Cfg<cute::SM90_64x64x16_F32F16F16_RS<kK, kMN>, half_t, half_t, float,
                    KSw32<half_t>, MNSw32<half_t>, 64, 64, true, EType::F16, EType::F16, EType::F32>;
using RSF16KK = Cfg<cute::SM90_64x64x16_F16F16F16_RS<kK, kK>, half_t, half_t, half_t,
                    KInter<half_t>, KSw64<half_t>, 64, 64, true, EType::F16, EType::F16, EType::F16>;
// imm-scale-a / imm-scale-b = -1.
using NegA16 = Cfg<cute::SM90_64x64x16_F32F16F16_SS<kK, kK, kNeg, kOne>, half_t, half_t, float,
                   KSw128<half_t>, KSw128<half_t>, 64, 64, false, EType::F16, EType::F16, EType::F32,
                   true, false>;
using NegB16 = Cfg<cute::SM90_64x64x16_F32BF16BF16_SS<kMN, kMN, kOne, kNeg>, bfloat16_t, bfloat16_t,
                   float, MNSw128<bfloat16_t>, MNSw128<bfloat16_t>, 64, 64, false, EType::BF16,
                   EType::BF16, EType::F32, false, true>;
using NegTF32 = Cfg<cute::SM90_64x64x8_F32TF32TF32_SS_TN<kNeg, kNeg>, tfloat32_t, tfloat32_t, float,
                    KSw128<tfloat32_t>, KSw128<tfloat32_t>, 64, 32, false, EType::TF32, EType::TF32,
                    EType::F32, true, true>;
using NegE4M3 = Cfg<cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<kNeg, kOne>, float_e4m3_t, float_e4m3_t,
                    float, KSw128<float_e4m3_t>, KSw128<float_e4m3_t>, 64, 128, false, EType::E4M3,
                    EType::E4M3, EType::F32, true, false>;
using NegRS = Cfg<cute::SM90_64x64x16_F32F16F16_RS<kK, kMN, kNeg, kNeg>, half_t, half_t, float,
                  KSw128<half_t>, MNSw64<half_t>, 64, 64, true, EType::F16, EType::F16, EType::F32,
                  true, true>;
// clang-format on

}  // namespace

int main() {
  e2e::Harness h("wgmma.cute");
  const e2e::Kern probe = {cute_probe_kernel,
                           "mode_probe",
                           8,
                           EType::F16,
                           EType::F16,
                           EType::F32,
                           false,
                           false,
                           false,
                           false,
                           false,
                           false,
                           false,
                           0};
  h.probe(probe);

  run_cfg<F16KK128>(h, "f32_f16_kk_sw128", 0xC0700100ull);
  run_cfg<F16KK64>(h, "f32_f16_kk_sw64", 0xC0700200ull);
  run_cfg<F16KK32>(h, "f32_f16_kk_sw32", 0xC0700300ull);
  run_cfg<F16KKI>(h, "f32_f16_kk_inter", 0xC0700400ull);
  run_cfg<F16MM128>(h, "f32_f16_mm_sw128", 0xC0700500ull);
  run_cfg<F16MM64>(h, "f32_f16_mm_sw64", 0xC0700600ull);
  run_cfg<F16MM32>(h, "f32_f16_mm_sw32", 0xC0700700ull);
  run_cfg<F16MMI>(h, "f32_f16_mm_inter", 0xC0700800ull);
  run_cfg<F16N8>(h, "f32_f16_n8_kk_inter", 0xC0700900ull);
  run_cfg<F16N256>(h, "f32_f16_n256_kk_sw128", 0xC0700A00ull);
  run_cfg<H16KM>(h, "f16_f16_km_sw64_sw32", 0xC0700B00ull);
  run_cfg<H16MK>(h, "f16_f16_mk_sw128_inter", 0xC0700C00ull);
  run_cfg<BF16KK>(h, "f32_bf16_kk_sw128_sw32", 0xC0700D00ull);
  run_cfg<BF16MM>(h, "f32_bf16_mm_sw64_inter", 0xC0700E00ull);
  run_cfg<TF32A>(h, "f32_tf32_kk_sw128_sw64", 0xC0700F00ull);
  run_cfg<TF32B>(h, "f32_tf32_kk_sw32_inter", 0xC0701000ull);
  run_cfg<E4M3A>(h, "f32_e4m3_kk_sw128_sw32", 0xC0701100ull);
  run_cfg<E4M3B>(h, "f32_e4m3_kk_inter_sw64", 0xC0701200ull);
  run_cfg<S8A>(h, "s32_s8_kk_sw128_sw64", 0xC0701300ull);
  run_cfg<S8Sat>(h, "s32_s8_satfinite_kk_sw32_inter", 0xC0701400ull);
  run_cfg<RSF32KK>(h, "rs_f32_f16_kk_sw128", 0xC0701500ull);
  run_cfg<RSF32KM>(h, "rs_f32_f16_km_sw32", 0xC0701600ull);
  run_cfg<RSF16KK>(h, "rs_f16_f16_kk_inter_sw64", 0xC0701700ull);
  run_cfg<NegA16>(h, "neg_a_f32_f16_kk_sw128", 0xC0701800ull);
  run_cfg<NegB16>(h, "neg_b_f32_bf16_mm_sw128", 0xC0701900ull);
  run_cfg<NegTF32>(h, "neg_ab_f32_tf32_kk_sw128", 0xC0701A00ull);
  run_cfg<NegE4M3>(h, "neg_a_f32_e4m3_kk_sw128", 0xC0701B00ull);
  run_cfg<NegRS>(h, "neg_ab_rs_f32_f16_km_sw128_sw64", 0xC0701C00ull);
  return h.finish();
}
