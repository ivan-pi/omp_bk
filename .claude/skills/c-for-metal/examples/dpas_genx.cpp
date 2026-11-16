// dpas_genx.cpp -- one XMX (systolic) instruction: D(8x8, float) = A(8x16, half) * B(16x8, half)
//
// cm_dpas<Src1Prec, Src2Prec, SystolicDepth, RepeatCount>(Acc, Src1, Src2)
//   Src1 = B in VNNI layout, Src2 = A row-major, both packed into int vectors.
//   N (columns of B/C/D) = ExecSize: 8 on XeHPG (DG2/Arc A-series), 16 on
//   XeHPC (PVC) and Xe2.
//   K = SystolicDepth (8) * OpsPerChannel (2 for half/bf16, 4 for int8, 8 for int4/int2).
//   M = RepeatCount (1..8).
//
// Build for a GPU with XMX (CM_HAS_DPAS): cm_build.sh -d dg2 dpas_genx.cpp
// (the 2021 cmc needs -mcpu=XEHP_SDV instead: references/legacy-sdk.md).
#include <cm/cm.h>

constexpr int M = 8;          // RepeatCount
constexpr int N = 8;          // ExecSize on DG2
constexpr int K = 8 * 2;      // SystolicDepth * OpsPerChannel(half)

extern "C" _GENX_MAIN_ void dpas_hf(SurfaceIndex A [[type("buffer_t")]], // M x K half, row-major
                                    SurfaceIndex B [[type("buffer_t")]], // K x N half, VNNI
                                    SurfaceIndex D [[type("buffer_t")]]) // M x N float
{
  static_assert(M * K / 2 == 64 && K * N / 2 == 64, "sizes below assume 64 dwords");
  // 256 bytes each: two OWord block reads of 128 bytes (the per-message maximum).
  vector<int, M * K / 2> a; // two halves per dword
  vector<int, K * N / 2> b; // VNNI: dword (k/2, n) holds B(k, n) | B(k+1, n) << 16
  read(A, 0, a.select<32, 1>(0));
  read(A, 128, a.select<32, 1>(32));
  read(B, 0, b.select<32, 1>(0));
  read(B, 128, b.select<32, 1>(32));

  vector<float, M * N> acc = 0.0f;
  acc = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, M>(acc, b, a);

  write(D, 0, acc.select<32, 1>(0));
  write(D, 128, acc.select<32, 1>(32));
}
