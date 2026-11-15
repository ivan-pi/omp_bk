// bk1_offload.cpp -- OpenMP offload harness for BK1 kernels on the GPU (the kernel
// variants live in openmp/offload/bk1_offload/*.h, one header per strategy): one order
// per build (-DNQ=n, nq = p + 2), all kernel variants in one binary, each
// timed with the drivers' harness and checked against SumFactorization.
//
//   scripts/build.sh bk1_offload_<p>       (-> build/a770/bk1_offload_<p>)
//   scripts/run.sh bk1_offload_<p> [nelmt] [ntests]
//   BK_KERNEL="teams teams1p" ...             run a subset (default: all)
//   BK_REF=0 ...                              skip the (slow) reference kernel;
//                                             the first variant is the reference
//
// Data: pseudo-random (bk::fill_random_test_data).  The check prints the
// largest difference to the reference relative to the largest |reference|.

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk1_sumfact.h"
#include "bk1_gpu.h"
#include "bk1_offload/teams.h"
#include "bk1_offload/plane.h"
#include "bk1_offload/plane_tid.h"
#include "bk1_offload/grid_stride.h"
#include "bk1_offload/grid_inline.h"

#ifndef NQ
#define NQ 6
#endif

using namespace bk;

using namespace bk1gpu;

namespace {

struct variant {
    const char* name;
    sumfact_kernel<float> kernel;
    bool jxw_transposed = false;   // kernel reads JxW stored as (e, r, q, p)
};

// Fixed batches go through cap_batch (bk1_gpu.h): every variant shares one
// device image, and one that does not fit the SLM or 1024 threads at a high
// order keeps the whole image from loading.
const variant variants[] = {
    {"ref",     &SumFactorization<float, NQ>},
    {"teams",   &SumFactorizationTeams<float, NQ>},
    {"teams1p", &SumFactorizationTeams1P<float, NQ>},
    {"plane1",  &SumFactorizationPlane<float, NQ, 1>},
    {"plane128", &SumFactorizationPlane<float, NQ, plane_batch(NQ, 128)>},
    {"plane256", &SumFactorizationPlane<float, NQ, plane_batch(NQ, 256)>},
    {"planeC1",   &SumFactorizationPlaneC<float, NQ, 1>},
    {"planeC128", &SumFactorizationPlaneC<float, NQ, plane_batch(NQ, 128)>},
    {"planeC256", &SumFactorizationPlaneC<float, NQ, plane_batch(NQ, 256)>},
    {"planeC512", &SumFactorizationPlaneC<float, NQ, plane_batch(NQ, 512)>},
    {"planeV1",   &SumFactorizationPlaneV<float, NQ, 1>},
    {"planeV128", &SumFactorizationPlaneV<float, NQ, plane_batch(NQ, 128)>},
    {"planeV256", &SumFactorizationPlaneV<float, NQ, plane_batch(NQ, 256)>},
    {"planeVP1",   &SumFactorizationPlaneV<float, NQ, 1, 1>},
    {"planeVP128", &SumFactorizationPlaneV<float, NQ, plane_batch(NQ, 128), 1>},
    {"planeG1",   &SumFactorizationPlaneG<float, NQ, 1>},
    {"planeG128", &SumFactorizationPlaneG<float, NQ, plane_batch(NQ, 128)>},
    {"gsR1",    &GridStrideRaw<float, NQ, 1, gs_block(NQ, 1)>},
    {"gsR128",  &GridStrideRaw<float, NQ, plane_batch(NQ, 128), gs_block(NQ, plane_batch(NQ, 128))>},
    {"gsW1",    &GridStrideWS<float, NQ, 1, gs_block(NQ, 1)>},
    {"gsW128",  &GridStrideWS<float, NQ, plane_batch(NQ, 128), gs_block(NQ, plane_batch(NQ, 128))>},
    {"gsW_b2",  &GridStrideWS<float, NQ, cap_batch(NQ, 2), gs_block(NQ, cap_batch(NQ, 2))>},
    {"gsW_b3",  &GridStrideWS<float, NQ, cap_batch(NQ, 3), gs_block(NQ, cap_batch(NQ, 3))>},
    {"gsW_b4",  &GridStrideWS<float, NQ, cap_batch(NQ, 4), gs_block(NQ, cap_batch(NQ, 4))>},
    {"gsW_b6",  &GridStrideWS<float, NQ, cap_batch(NQ, 6), gs_block(NQ, cap_batch(NQ, 6))>},
    {"gsW_b8",  &GridStrideWS<float, NQ, cap_batch(NQ, 8), gs_block(NQ, cap_batch(NQ, 8))>},
    {"gsI_b1",  &GridStrideInline<float, NQ, 1, gs_block(NQ, 1)>, true},
    {"gsI_b4",  &GridStrideInline<float, NQ, cap_batch(NQ, 4), gs_block(NQ, cap_batch(NQ, 4))>, true},
    {"gsI_b8",  &GridStrideInline<float, NQ, cap_batch(NQ, 8), gs_block(NQ, cap_batch(NQ, 8))>, true},
    {"gsI_b16", &GridStrideInline<float, NQ, cap_batch(NQ, 16), gs_block(NQ, cap_batch(NQ, 16))>, true},
    {"gsI_b32", &GridStrideInline<float, NQ, cap_batch(NQ, 32), gs_block(NQ, cap_batch(NQ, 32))>, true},
    {"gsI_b64", &GridStrideInline<float, NQ, cap_batch(NQ, 64), gs_block(NQ, cap_batch(NQ, 64))>, true},
    {"gsF_f128",  &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 128)), 128, false>, true},
    {"gsFP_f128", &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 128)), 128, true>, true},
    {"gsF_f256",  &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 256)), 256, false>, true},
    {"gsFP_f256", &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 256)), 256, true>, true},
    {"gsF_f512",  &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 512)), 512, false>, true},
    {"gsFP_f512", &GridStridePadded<float, NQ, cap_batch(NQ, fill_batch(NQ, 512)), 512, true>, true},
    {"gsP_b1",  &GridStridePadded<float, NQ, cap_batch(NQ, 1), gs_block(NQ, cap_batch(NQ, 1)), true>, true},
    {"gsP_b4",  &GridStridePadded<float, NQ, cap_batch(NQ, 4), gs_block(NQ, cap_batch(NQ, 4)), true>, true},
    {"gsP_b8",  &GridStridePadded<float, NQ, cap_batch(NQ, 8), gs_block(NQ, cap_batch(NQ, 8)), true>, true},
    {"gsP_b16",  &GridStridePadded<float, NQ, cap_batch(NQ, 16), gs_block(NQ, cap_batch(NQ, 16)), true>, true},
    {"gsT_b1",  &GridStrideJt<float, NQ, 1, gs_block(NQ, 1)>, true},
    {"gsT_b4",  &GridStrideJt<float, NQ, cap_batch(NQ, 4), gs_block(NQ, cap_batch(NQ, 4))>, true},
    {"gsT_b8",  &GridStrideJt<float, NQ, cap_batch(NQ, 8), gs_block(NQ, cap_batch(NQ, 8))>, true},
    {"gsJ_b1",  &GridStrideJxW<float, NQ, 1, gs_block(NQ, 1)>},
    {"gsJ_b2",  &GridStrideJxW<float, NQ, cap_batch(NQ, 2), gs_block(NQ, cap_batch(NQ, 2))>},
    {"gsJ_b4",  &GridStrideJxW<float, NQ, cap_batch(NQ, 4), gs_block(NQ, cap_batch(NQ, 4))>},
    {"gsD1",    &GridStrideDirect<float, NQ, 1, gs_block(NQ, 1)>},
    {"gsD128",  &GridStrideDirect<float, NQ, plane_batch(NQ, 128), gs_block(NQ, plane_batch(NQ, 128))>},
    {"teamsD",  &SumFactorizationTeamsD<float, NQ>},
    {"planeL1",   &SumFactorizationPlaneL<float, NQ, 1>},
    {"planeL128", &SumFactorizationPlaneL<float, NQ, plane_batch(NQ, 128)>},
    {"planeT1",   &SumFactorizationPlaneTid<float, NQ, 1, false>},
    {"planeT256", &SumFactorizationPlaneTid<float, NQ, plane_batch(NQ, 256), false>},
    {"planeX1",   &SumFactorizationPlaneTid<float, NQ, 1, true>},
    {"planeX256", &SumFactorizationPlaneTid<float, NQ, plane_batch(NQ, 256), true>},
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 19;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    const bool with_ref = get_env("BK_REF").value_or("1") != "0";
    g_grid = env_int("BK_TEAMS", 0);

    const std::array<float, nm * nq> basis = make_test_basis<float, nm, nq>();
    std::vector<float> JxW(nelmt * nq * nq * nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    fill_random_test_data(in, JxW);
    // JxW transposed per element, only if a selected kernel reads it so
    bool need_jxw_t = false;
    for (const variant& v : variants) {
        need_jxw_t = need_jxw_t || (v.jxw_transposed && selected(v.name));
    }
    const std::vector<float> JxW_t = need_jxw_t ? transpose_jxw(JxW, nelmt, nq) : std::vector<float>();
    std::vector<float> ref(in.size());
    std::vector<float> out(in.size());
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());

    std::printf("BK1 offload: p = %d, nq = %d, nelmt = %zu, ntests = %d, team rounding %d\n",
                nq - 2, nq, nelmt, ntests, TEAM_ROUND);

    // The arrays stay on the device across the variants: the inputs are
    // uploaded once; each variant's output is zeroed on the device (so an
    // entry a kernel misses shows in maxdiff) and copied back after its runs.
    // The first kernel run (ref, unless BK_REF=0) is the reference of the others.
    const float* d_basis = basis.data();
    const float* d_jxw = JxW.data();
    const float* d_jxw_t = JxW_t.data();
    const float* d_in = in.data();
    float* d_ref = ref.data();
    float* d_out = out.data();
    const std::size_t n_in = in.size();
    const std::size_t n_jxw = JxW.size();
    const std::size_t n_jxw_t = JxW_t.size();
    bool have_ref = false;
    bool invalid = false;   // no early return: the target data block must end normally
    double big = 0.0;
    #pragma omp target data map(to: d_basis[:nm*nq], d_jxw[:n_jxw], d_jxw_t[:n_jxw_t], d_in[:n_in]) \
        map(alloc: d_ref[:n_in], d_out[:n_in])
    {
        for (const variant& v : variants) {
            const bool is_ref = std::strcmp(v.name, "ref") == 0;
            if (is_ref && !with_ref) {
                continue;
            }
            if (!is_ref && !selected(v.name)) {
                continue;
            }
            float* dst = have_ref ? d_out : d_ref;
            #pragma omp target teams distribute parallel for map(alloc: dst[:n_in])
            for (std::size_t i = 0; i < n_in; ++i) {
                dst[i] = 0.0f;
            }
            const float* jxw = v.jxw_transposed ? d_jxw_t : d_jxw;
            const double t = time_min(ntests, [&] { v.kernel(nelmt, d_basis, jxw, d_in, dst); });
            #pragma omp target update from(dst[:n_in])
            if (g_short_team != 0) {
                std::printf("%-10s a team got fewer threads than its work items; result invalid\n", v.name);
                invalid = true;
                break;
            }
            double diff = 0.0;
            if (have_ref) {
                for (std::size_t i = 0; i < n_in; ++i) {
                    diff = std::max(diff, double(std::fabs(out[i] - ref[i])));
                }
            } else {
                for (const float r : ref) {
                    big = std::max(big, double(std::fabs(r)));
                }
            }
            std::printf("%-10s GDoF/s = %8.3f  GB/s = %7.1f  time = %8.3f ms  maxdiff = %.2e%s\n",
                        v.name, 1e-9 * n_in / t, 1e-9 * bytes / t, 1e3 * t,
                        have_ref ? diff / big : 0.0, have_ref ? "" : " (reference)");
            have_ref = true;
        }
    }
    return invalid ? 1 : 0;
}
