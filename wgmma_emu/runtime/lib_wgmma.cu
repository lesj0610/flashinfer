// wgmma library entry points (PLAN section 6). Owner: W.
// Included by runtime/emu_lib.cu (library PTX) and directly by tests/wgmma/test_direct.cu.
// Signatures and the variant / ids / table formats are documented at the top of
// include/wgmma_emu/wgmma_device.cuh.
#include "wgmma_emu/wgmma_device.cuh"

extern "C" {

__device__ __noinline__ void wgemu_wgmma(uint64_t d_ptr, uint64_t adesc, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint64_t bdesc, uint32_t meta,
                                         uint32_t scale_d, uint32_t variant, uint32_t ids,
                                         uint64_t pend) {
  wgemu::wg::issue(d_ptr, adesc, a0, a1, a2, a3, bdesc, meta, scale_d, variant, ids, pend);
}

__device__ __noinline__ void wgemu_wgmma_check_init(uint64_t pend, uint32_t n_acc, uint32_t n_a,
                                                    uint32_t flags) {
  wgemu::wg::check_init(pend, n_acc, n_a, flags);
}

__device__ __noinline__ void wgemu_wgmma_fence(uint64_t pend) { wgemu::wg::check_fence(pend); }

__device__ __noinline__ void wgemu_wgmma_commit(uint64_t pend) { wgemu::wg::check_commit(pend); }

__device__ __noinline__ void wgemu_wgmma_wait(uint64_t pend, uint32_t n) {
  wgemu::wg::check_wait(pend, n);
}

__device__ __noinline__ uint32_t wgemu_wgmma_fetch(uint64_t pend, uint32_t acc_id, uint64_t d_ptr) {
  return wgemu::wg::check_fetch(pend, acc_id, d_ptr);
}

__device__ __noinline__ void wgemu_wgmma_check_a(uint64_t pend, uint32_t a_id, uint32_t a0,
                                                 uint32_t a1, uint32_t a2, uint32_t a3) {
  const uint32_t a[4] = {a0, a1, a2, a3};
  wgemu::wg::check_a_regs(pend, a_id, a);
}

__device__ __noinline__ void wgemu_wgmma_check_acc(uint64_t pend, uint32_t acc_id, uint32_t kind) {
  wgemu::wg::check_acc_access(pend, acc_id, kind);
}

}  // extern "C"
