#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
#include <cute/numeric/complex.hpp>

namespace cute {
using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

struct CDNA2_32x32x2_F32F32F32F32_TN
{
  using DRegisters = float16[1];
  using ARegisters = float[1];
  using BRegisters = float[1];
  using CRegisters = float16[1];

  CUTE_HOST_DEVICE static void
  fma(float16         & d0,
      float const& a0,
      float const& b0,
      float16 const   & c0)
  {
    d0 = __builtin_amdgcn_mfma_f32_32x32x2f32(a0, b0, c0, 0, 0, 0);
  }
};
}