// bk1_offload.cpp -- OpenMP offload harness for BK1 kernels on the GPU (the kernel
// variants live in openmp/offload/bk1_offload/*.h, one header per strategy): one order
// per build (-DNQ=n, nq = p + 2), all kernel variants in one binary, each
// timed with the drivers' harness and checked against SumFactorization.
//
//   scripts/build.sh bk1_offload_6         (-> build/a770/bk1_offload_6)
//   scripts/run.sh bk1_offload_6 [nelmt] [ntests]
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
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"
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

// Fixed batches go through cap_batch (common.h): every variant shares one
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
    {"gsW_b1",  &GridStrideWS<float, NQ, 1, gs_block(NQ, 1)>},
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

bool selected(const char* name, const std::string& list)
{
    if (list.empty()) {
        return true;
    }
    std::istringstream ss(list);
    std::string w;
    while (ss >> w) {
        if (w == name) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 19;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    const std::string list = get_env("BK_KERNEL").value_or("");
    const bool with_ref = get_env("BK_REF").value_or("1") != "0";
    g_grid = std::atoi(get_env("BK_TEAMS").value_or("0").c_str());

    const std::array<float, nm * nq> basis = make_test_basis<float, nm, nq>();
    std::vector<float> JxW(nelmt * nq * nq * nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    fill_random_test_data(in, JxW);
    // JxW transposed per element, (e, r, q, p), for the kernels that read it so
    std::vector<float> JxW_t(JxW.size());
    for (std::size_t e = 0; e < nelmt; ++e) {
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    JxW_t[((e * nq + r) * nq + q) * nq + p] = JxW[((e * nq + p) * nq + q) * nq + r];
                }
            }
        }
    }
    std::vector<float> ref(in.size());
    std::vector<float> out(in.size());
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());

    std::printf("BK1 offload: p = %d, nq = %d, nelmt = %zu, ntests = %d, team rounding %d\n",
                nq - 2, nq, nelmt, ntests, TEAM_ROUND);
    bool have_ref = false;
    for (const variant& v : variants) {
        const bool is_ref = std::strcmp(v.name, "ref") == 0;
        if (is_ref && !with_ref) {
            continue;
        }
        if (!is_ref && !selected(v.name, list)) {
            continue;
        }
        std::vector<float>& dst = have_ref ? out : ref;
        std::fill(dst.begin(), dst.end(), 0.0f);
        const double t = time_sumfact<float, nq>(nelmt, ntests, basis.data(),
                                                 v.jxw_transposed ? JxW_t.data() : JxW.data(),
                                                 in.data(), dst.data(), v.kernel);
        if (g_short_team != 0) {
            std::printf("%-10s a team got fewer threads than its work items; result invalid\n", v.name);
            return 1;
        }
        double diff = 0.0;
        double big = 0.0;
        if (have_ref) {
            for (std::size_t i = 0; i < ref.size(); ++i) {
                diff = std::max(diff, double(std::fabs(out[i] - ref[i])));
                big = std::max(big, double(std::fabs(ref[i])));
            }
        }
        std::printf("%-10s GDoF/s = %8.3f  GB/s = %7.1f  time = %8.3f ms  maxdiff = %.2e%s\n",
                    v.name, 1e-9 * in.size() / t, 1e-9 * bytes / t, 1e3 * t,
                    have_ref ? diff / big : 0.0, have_ref ? "" : " (reference)");
        have_ref = true;
    }
    return 0;
}
