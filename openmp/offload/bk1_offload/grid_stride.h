// grid_stride.h -- the CUDA/Kokkos kernel structure: seven steps through SLM
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_GRID_STRIDE_H
#define BK1_OFFLOAD_GRID_STRIDE_H

#include <omp.h>

#include <cstddef>

#include "bk_common.h"
#include "bk1_sumfact.h"
#include "bk1_gpu.h"
#include "bk1_gpu_steps.h"

namespace bk1gpu {

using bk::ndview;

// ---------------------------------------------------------------------------
// gsR / gsW: the grid-strided CUDA kernel (BK1::Parallel::MassOperator) in
// OpenMP.  A fixed grid of teams strides over batches of elements; each
// batch is copied contiguously from global to SLM (step 1), runs seven
// contraction steps in SLM with each thread holding one input column in
// registers, and is copied back contiguously (step 9).  The step bodies are
// shared; the variants differ only in how threads are launched:
//   gsR: teams num_teams(G) + parallel, thread-stride loops over
//        omp_get_thread_num() and `omp barrier` (a literal CUDA port), the
//        basis copied to SLM once per team;
//   gsW: teams distribute dist_schedule(static, 1) num_teams(G) over the
//        batches (the same grid stride) + an `omp for` per step.
// BK_TEAMS=G sets the grid (default 0: one team per batch).
// ---------------------------------------------------------------------------
inline int g_grid = 0;


template <typename T, int nq, int batch, int block>
void GridStrideRaw(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    using S = gs_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams num_teams(nteams) thread_limit(block)
    {
        T s_basis[nm * nq];
        T s_wsp0[batch * nq3];
        T s_wsp1[batch * nq3];
        const int team = omp_get_team_num();
        const int nt = omp_get_num_teams();

        #pragma omp parallel
        {
            const int tid0 = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            for (int i = tid0; i < nm * nq; i += nth) {
                s_basis[i] = basis[i];
            }
            #pragma omp barrier

            for (int eb = team; eb < nbatches; eb += nt) {
                const int remaining = ne - eb * batch;
                const int cb = (batch < remaining) ? batch : remaining;
                const T* b_in = in + std::size_t(eb) * (batch * nm3);
                const T* b_JxW = JxW + std::size_t(eb) * (batch * nq3);
                T* b_out = out + std::size_t(eb) * (batch * nm3);

                for (int i = tid0; i < cb * nm3; i += nth) {
                    s_wsp0[i] = b_in[i];
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nm * nm; t += nth) {
                    S::s2(t, s_basis, s_wsp0, s_wsp1);
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nq * nm; t += nth) {
                    S::s3(t, s_basis, s_wsp1, s_wsp0);
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nq * nq; t += nth) {
                    S::s4(t, s_basis, b_JxW, s_wsp0, s_wsp1);
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nq * nq; t += nth) {
                    S::s6(t, s_basis, s_wsp1, s_wsp0);
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nm * nq; t += nth) {
                    S::s7(t, s_basis, s_wsp0, s_wsp1);
                }
                #pragma omp barrier
                for (int t = tid0; t < cb * nm * nm; t += nth) {
                    S::s8(t, s_basis, s_wsp1, s_wsp0);
                }
                #pragma omp barrier
                for (int i = tid0; i < cb * nm3; i += nth) {
                    b_out[i] = s_wsp0[i];
                }
                #pragma omp barrier
            }
        }
    }
}

template <typename T, int nq, int batch, int block>
void GridStrideWS(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    using S = gs_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute num_teams(nteams) thread_limit(block) dist_schedule(static, 1)
    for (int eb = 0; eb < nbatches; ++eb) {
        T s_wsp0[batch * nq3];
        T s_wsp1[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                s_wsp0[i] = b_in[i];
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s2(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                S::s3(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s4(t, basis, b_JxW, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s6(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                S::s7(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s8(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                b_out[i] = s_wsp0[i];
            }
        }
    }
}

// gsT: gsW reading JxW stored transposed per element, (r, q, p) as the CUDA
// and Kokkos kernels index it; the harness passes a transposed copy
template <typename T, int nq, int batch, int block>
void GridStrideJt(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    using S = gs_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute num_teams(nteams) thread_limit(block) dist_schedule(static, 1)
    for (int eb = 0; eb < nbatches; ++eb) {
        T s_wsp0[batch * nq3];
        T s_wsp1[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                s_wsp0[i] = b_in[i];
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s2(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                S::s3(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s4t(t, basis, b_JxW, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s6(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                S::s7(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s8(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                b_out[i] = s_wsp0[i];
            }
        }
    }
}

// gsJ: gsW with each batch's JxW also copied contiguously into SLM in step 1
// (step 4's JxW reads are nq * nq apart between neighbouring lanes)
template <typename T, int nq, int batch, int block>
void GridStrideJxW(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    using S = gs_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute num_teams(nteams) thread_limit(block) dist_schedule(static, 1)
    for (int eb = 0; eb < nbatches; ++eb) {
        T s_wsp0[batch * nq3];
        T s_wsp1[batch * nq3];
        T s_jxw[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                s_wsp0[i] = b_in[i];
            }
            #pragma omp for nowait
            for (int i = 0; i < cb * nq3; ++i) {
                s_jxw[i] = b_JxW[i];
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s2(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                S::s3(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s4(t, basis, s_jxw, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s6(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                S::s7(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s8(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                b_out[i] = s_wsp0[i];
            }
        }
    }
}

// gsD: gsW without the copy steps -- step 2 reads in and step 8 writes out
// directly (both have the (e, i, j, k) layout of the SLM box they replace)
template <typename T, int nq, int batch, int block>
void GridStrideDirect(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    using S = gs_steps<T, nq>;
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute num_teams(nteams) thread_limit(block) dist_schedule(static, 1)
    for (int eb = 0; eb < nbatches; ++eb) {
        // wsp0 holds only the outputs of steps 3 and 6 (nq * nq * nm per
        // element); in and out are read and written by steps 2 and 8 directly
        T s_wsp0[batch * nq * nq * nm];
        T s_wsp1[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s2(t, basis, b_in, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                S::s3(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s4(t, basis, b_JxW, s_wsp0, s_wsp1);
            }
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                S::s6(t, basis, s_wsp1, s_wsp0);
            }
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                S::s7(t, basis, s_wsp0, s_wsp1);
            }
            #pragma omp for nowait
            for (int t = 0; t < cb * nm * nm; ++t) {
                S::s8(t, basis, s_wsp1, b_out);
            }
        }
    }
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_GRID_STRIDE_H
