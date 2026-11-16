// specconst_genx.cpp -- SPIR-V specialization constants in C for Metal.
//
// cm_spec_constant<T, ID, Default>() (cm/cm_spec_constant.h, included by
// cm/cm.h) becomes OpSpecConstant ID in the SPIR-V that cmc emits. The value
// is fixed when the SPIR-V is finalized, not when cmc runs:
//   OpenCL       cl::Program::setSpecializationConstant(ID, value) before build()
//   Level Zero   ze_module_constants_t in ze_module_desc_t::pConstants
//   ocloc        -spec_const file.txt   (newer ocloc; lines "ID: value", floats as bit patterns)
// IGC then folds it like a literal: branches on it disappear, loops on it
// get a constant trip count. It is NOT a C++ constant expression, so it can't
// size a vector, be a template argument or feed `if constexpr` -- for those,
// use -D at cmc time or template instantiations (one extern "C" kernel each).
//
// Types: integral (any default) and float/double (the default is always 0).
//
// Needs intel-igc-cm's cmc (tested with 1.0.225) and its matched IGC to set values
// at run time (the 2021 SDK cmc crashes on cm_spec_constant). Tested: PASS
// on an Arc A770 and a UHD 750 (run_examples.sh).
//
// Kernel: y = SCALE * x on BLOCKS*32 floats per thread (BLOCKS <= MAXB).
#include <cm/cm.h>

constexpr int BLOCK = 32; // floats per OWord block message
constexpr int MAXB = 8;   // register budget: a compile-time upper bound

extern "C" _GENX_MAIN_ void scale(SurfaceIndex x [[type("buffer_t")]],
                                  SurfaceIndex y [[type("buffer_t")]]) {
  const int blocks = cm_spec_constant<int, 0, 4>(); // id 0, default 4
  const float s = cm_spec_constant<float, 1>();     // id 1, default 0.0f
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint off = tid * blocks * BLOCK * sizeof(float);

  vector<float, MAXB * BLOCK> v;
#pragma unroll
  for (int i = 0; i < MAXB; ++i) {
    if (i < blocks) { // folded after specialization
      read(x, off + i * BLOCK * sizeof(float), v.select<BLOCK, 1>(i * BLOCK));
    }
  }
  v *= s;
#pragma unroll
  for (int i = 0; i < MAXB; ++i) {
    if (i < blocks) {
      write(y, off + i * BLOCK * sizeof(float), v.select<BLOCK, 1>(i * BLOCK));
    }
  }
}
