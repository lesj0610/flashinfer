// Shared basic types for the SM80 wgmma emulation layer (fork-only).
// Owned by the integration thread; other modules include this, never edit it.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define WGEMU_HD __host__ __device__ inline
#define WGEMU_HDF __host__ __device__ __forceinline__
#define WGEMU_DEV __device__ inline
#else
#define WGEMU_HD inline
#define WGEMU_HDF inline
#endif

namespace wgemu {

// Element types. Values are part of the variant encoding used by the
// translator, so they must never change.
enum class EType : uint8_t {
  F16 = 0,
  BF16 = 1,
  TF32 = 2,
  E4M3 = 3,
  E5M2 = 4,
  S8 = 5,
  U8 = 6,
  B1 = 7,
  F32 = 8,
  S32 = 9,
};

WGEMU_HD int etype_bits(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 16;
    case EType::TF32:
    case EType::F32:
    case EType::S32:
      return 32;
    case EType::E4M3:
    case EType::E5M2:
    case EType::S8:
    case EType::U8:
      return 8;
    case EType::B1:
      return 1;
  }
  return 0;
}

WGEMU_HD bool etype_is_fp8(EType t) { return t == EType::E4M3 || t == EType::E5M2; }
WGEMU_HD bool etype_is_int8(EType t) { return t == EType::S8 || t == EType::U8; }
WGEMU_HD bool etype_is_float_input(EType t) {
  return t == EType::F16 || t == EType::BF16 || t == EType::TF32 || etype_is_fp8(t);
}

}  // namespace wgemu
