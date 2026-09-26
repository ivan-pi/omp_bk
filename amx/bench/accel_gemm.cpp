// accel_gemm.cpp -- Apple Accelerate sgemm as a calibration point for the AMX
// unit: (1) large square GEMM = what a vendor-tuned AMX kernel sustains with
// memory operands, in GFLOP/s, to compare with amx_peak from bw_test;
// (2) the small shapes of one BK1 contraction, to show why a BLAS call per
// tile is not a usable implementation (per-call overhead vs ~1 ns of work).
//
//   make accel_gemm    (macOS only: links the Accelerate framework)
//   ./accel_gemm
//
// Output CSV: shape,M,N,K,calls,seconds,GFLOP_per_s,ns_per_call
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>

using clk = std::chrono::high_resolution_clock;

static void bench(const char* label, int M, int N, int K, int calls, int reps) {
    std::vector<float> A(std::size_t(M) * K, 1.0f), B(std::size_t(K) * N, 0.5f), C(std::size_t(M) * N, 0.0f);
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        for (int c = 0; c < calls; ++c) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, A.data(), M, B.data(), K, 1.0f, C.data(), M);
        }
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    const double flops = 2.0 * M * N * K * double(calls);
    std::printf("%s,%d,%d,%d,%d,%.6f,%.2f,%.1f\n", label, M, N, K, calls, best,
                1e-9 * flops / best, 1e9 * best / calls);
}

int main() {
    std::printf("shape,M,N,K,calls,seconds,GFLOP_per_s,ns_per_call\n");
    // 1) large: the unit fed from memory by a tuned kernel (Accelerate uses
    //    all AMX units it can; compare with amx_peak from bw_test)
    bench("large", 2048, 2048, 2048, 1, 5);
    bench("large", 4096, 4096, 4096, 1, 3);
    // 2) one BK1 contraction per free-index combination, elements on M:
    //    C(16 x N) = X(16 x S) * B(S x N)   for p = 2, 8, 14  (S = nm, N = nq)
    bench("bk1_tile_p2", 16, 4, 3, 100000, 3);
    bench("bk1_tile_p8", 16, 10, 9, 100000, 3);
    bench("bk1_tile_p14", 16, 16, 15, 100000, 3);
    // 3) one whole step with all free combinations fused into N (only the
    //    first step of the chain has this layout): C(N x F*16) = B^T(N x S) X(S x F*16)
    bench("bk1_step_p8", 10, 81 * 16, 9, 10000, 3);
    bench("bk1_step_p14", 16, 225 * 16, 15, 10000, 3);
    return 0;
}
