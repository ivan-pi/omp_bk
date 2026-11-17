// bk3_offload.cpp -- BK3 (Poisson operator) kernel variants in OpenMP
// offload, one order per build (-DNQ=nq, nq = p + 2), every variant timed
// and checked against the double-precision CPU reference.  The kernels of
// sycl/bk3_sycl.cpp (steps in include/bk3_gpu_steps.h) with the work arrays
// at team scope, which icpx places in SLM, and an `omp for` per step.  (icpx
// 2026.1 rejects `omp allocate(...) allocator(omp_pteam_mem_alloc)` for
// spir64_gen.)
//
//   scripts/build.sh bk3_offload_<p>       (-> build/a770/bk3_offload_<p>)
//   scripts/run.sh bk3_offload_<p> [nelmt] [ntests]
//   BK_KERNEL="sf_b1 cu_b2" ...            run a subset (default: all; -name excludes)
//   BK_TEAMS=n                             cu: n teams stride over the batches
//
// Variants (b<batch>: elements per team):
//   ref           BK3.cpp's kernel (openmp/bk3_sumfact.h): one element per
//                 iteration of a teams loop, private work arrays
//   cu_b<batch>   port of the CUDA kernel
//   sf_b<batch>   register-column sweeps through three SLM boxes
//   suffix r      B and D held in registers by every thread instead of read
//                 from SLM (cu) or global memory (sf)
// A variant whose batch is clamped to an earlier one's is the same kernel and
// is reported as such instead of being run again.
//
// Data: pseudo-random in and G (bk::fill_random_test_data), G factor-major
// G(e, f, p, q, r).  GB/s counts in + out + G.

#include <omp.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk3_gpu_steps.h"
#include "bk3_sumfact.h"

#ifndef NQ
#define NQ 6
#endif

namespace {

inline int g_grid = 0;   // BK_TEAMS

template <typename T, int nq, int batch, bool breg>
void bk3_cu(const std::size_t nelmt, const T* __restrict__ basis, const T* __restrict__ dbasis,
            const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out)
{
    using interp = bk1gpu::gs_steps<T, nq>;
    using steps = bk3gpu::cu_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int nm3 = nm * nm * nm;
    constexpr int block = bk3gpu::cu_block(nq, batch);
    static_assert(block <= 1024, "a team holds at most 1024 threads");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;

    #pragma omp target teams num_teams(nteams) thread_limit(block) \
        map(to: basis[:nm*nq], dbasis[:nq2], G[:nelmt*6*nq3], in[:nelmt*nm3]) map(from: out[:nelmt*nm3])
    {
        T s_basis[nm * nq];
        T s_dbasis[nq2];
        T w0[batch * nq3];
        T w1[batch * nq3];
        T rqr[batch * nq3];
        T rqs[batch * nq3];
        T* rqt = w0;
        #pragma omp parallel
        {
            T Bv[nm * nq];
            T Dv[nq2];
            if constexpr (breg) {
                bk3gpu::load_basis<T, nq>(basis, dbasis, Bv, Dv);
            }
            const T* B = breg ? Bv : s_basis;
            const T* D = breg ? Dv : s_dbasis;
            #pragma omp for
            for (int t = 0; t < nm * nq; ++t) {
                s_basis[t] = basis[t];
            }
            #pragma omp for
            for (int t = 0; t < nq2; ++t) {
                s_dbasis[t] = dbasis[t];
            }
            for (int eb = omp_get_team_num(); eb < nbatches; eb += omp_get_num_teams()) {
                const int cb = std::min(batch, ne - eb * batch);
                const T* b_in = in + std::size_t(eb) * (batch * nm3);
                const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
                T* b_out = out + std::size_t(eb) * (batch * nm3);
                #pragma omp for
                for (int x = 0; x < cb * nm3; ++x) {
                    w0[x] = b_in[x];
                }
                #pragma omp for
                for (int t = 0; t < cb * nm * nm; ++t) {
                    interp::s2(t, B, w0, w1);
                }
                #pragma omp for
                for (int t = 0; t < cb * nq * nm; ++t) {
                    interp::s3(t, B, w1, w0);
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    interp::s4u(t, B, w0, w1);
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    steps::fluxes(t, D, b_G, w1, rqr, rqs, rqt);
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    steps::divergence(t, D, rqr, rqs, rqt, w1);
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    interp::s6(t, B, w1, w0);
                }
                #pragma omp for
                for (int t = 0; t < cb * nm * nq; ++t) {
                    interp::s7(t, B, w0, w1);
                }
                #pragma omp for
                for (int t = 0; t < cb * nm * nm; ++t) {
                    interp::s8(t, B, w1, w0);
                }
                #pragma omp for
                for (int x = 0; x < cb * nm3; ++x) {
                    b_out[x] = w0[x];
                }
            }
        }
    }
}

template <typename T, int nq, int batch, bool breg>
void bk3_sf(const std::size_t nelmt, const T* __restrict__ gB, const T* __restrict__ gD,
            const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out)
{
    using steps = bk3gpu::sf_steps<T, nq, false>;
    constexpr int nm = nq - 1;
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int nm3 = nm * nm * nm;
    constexpr int ebox = steps::ebox;
    constexpr int block = bk3gpu::sf_block(nq, batch);
    static_assert(3 * batch * ebox * int(sizeof(T)) <= 61440, "three SLM boxes within 60 KB");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);

    #pragma omp target teams distribute thread_limit(block) \
        map(to: gB[:nm*nq], gD[:nq2], G[:nelmt*6*nq3], in[:nelmt*nm3]) map(from: out[:nelmt*nm3])
    for (int eb = 0; eb < nbatches; ++eb) {
        T lm[3 * batch * ebox];
        T* U = lm;
        T* R = lm + batch * ebox;
        T* S = lm + 2 * batch * ebox;
        const int cb = std::min(batch, ne - eb * batch);
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            T Bv[nm * nq];
            T Dv[nq2];
            if constexpr (breg) {
                bk3gpu::load_basis<T, nq>(gB, gD, Bv, Dv);
            }
            const T* B = breg ? Bv : gB;
            const T* D = breg ? Dv : gD;
            T ucol[nq];   // unused: no carry here
            #pragma omp for
            for (int x = 0; x < cb * nm3; ++x) {
                steps::load(x, b_in, S);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                steps::s1(t, B, S, R);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                steps::s2(t, B, R, S);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                steps::template s3<false>(t, B, S, U, ucol);
            }
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
                steps::s4(t, cb, D, U, R, S);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                steps::template s5<false>(t, D, b_G, U, R, S, ucol);
            }
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
                steps::s6(t, cb, D, R, S);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                steps::template s7<false>(t, B, U, R, S, ucol);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                steps::s8(t, B, U, R);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                steps::s9(t, B, R, S);
            }
            #pragma omp for nowait
            for (int x = 0; x < cb * nm3; ++x) {
                steps::store(x, S, b_out);
            }
        }
    }
}

using kernel_fn = void (*)(std::size_t, const float*, const float*, const float*, const float*, float*);

struct variant {
    const char* name;
    kernel_fn fn;
};

#define CU(B, R, SUFFIX) {"cu_b" #B SUFFIX, &bk3_cu<float, NQ, bk3gpu::cu_batch(NQ, B), R>}
#define SF(B, R, SUFFIX) {"sf_b" #B SUFFIX, &bk3_sf<float, NQ, bk3gpu::sf_batch(NQ, B), R>}

const variant variants[] = {
#ifndef BK_NO_BASELINES
    // the driver's one-element-per-thread kernel: a baseline, 30-170x slower
    // than the best variants on the A770 (the check is the host reference)
    {"ref", &bk::SumFactorization<float, NQ>},
#endif
    CU(1, false, ""), CU(2, false, ""), CU(4, false, ""), CU(1, true, "r"), CU(2, true, "r"),
    SF(1, false, ""), SF(2, false, ""), SF(4, false, ""), SF(1, true, "r"), SF(2, true, "r"),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    g_grid = bk::env_int("BK_TEAMS", 0);

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    std::vector<float> G(nelmt * 6 * nq * nq * nq);
    bk::fill_random_test_data(in, G);   // in in [-1, 1], G = 1.5 + [-1, 1]
    // BK_CHECK=0: benchmark only, no reference
    const bk::reference_check check(in.size(), [&](std::vector<double>& ref) {
        bk::reference_bk3(nq, nelmt, basis, D, G, in, ref);
    });
    std::vector<float> out(in.size());
    std::printf("BK3 offload: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d\n", nq - 2, nq, nelmt,
                in.size(), ntests);

    // the arrays stay on the device across the variants; each variant's
    // output is zeroed on the device and copied back after its runs
    const float* d_basis = basis.data();
    const float* d_D = D.data();
    const float* d_G = G.data();
    const float* d_in = in.data();
    float* d_out = out.data();
    const std::size_t n_in = in.size();
    const std::size_t n_G = G.size();
    const double bytes = sizeof(float) * (2.0 * n_in + n_G);
    #pragma omp target data map(to: d_basis[:nm*nq], d_D[:nq*nq], d_G[:n_G], d_in[:n_in]) map(alloc: d_out[:n_in])
    {
        for (std::size_t v = 0; v < std::size(variants); ++v) {
            const variant& var = variants[v];
            if (!bk::selected(var.name)) {
                continue;
            }
            if (const variant* same = bk::same_kernel(variants, v)) {
                std::printf("%-10s = %s\n", var.name, same->name);
                continue;
            }
            #pragma omp target teams distribute parallel for
            for (std::size_t i = 0; i < n_in; ++i) {
                d_out[i] = 0.0f;
            }
            var.fn(nelmt, d_basis, d_D, d_G, d_in, d_out);   // warm-up
            // timed: ntests launches of the kernel, the best wall time
            const double t = bk::time_min(ntests, [&] {
                var.fn(nelmt, d_basis, d_D, d_G, d_in, d_out);
            });
            const std::string chk = check.text([&]() -> const std::vector<float>& {
                #pragma omp target update from(d_out[:n_in])
                return out;
            });
            std::printf("%-10s GDoF/s = %7.3f  GB/s = %6.1f  time = %8.3f ms  relerr = %s\n", var.name,
                        1e-9 * n_in / t, 1e-9 * bytes / t, 1e3 * t, chk.c_str());
        }
    }
    return 0;
}
