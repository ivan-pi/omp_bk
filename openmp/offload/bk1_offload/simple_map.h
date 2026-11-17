// simple_map.h -- smap: the "SimpleMap" kernel of the dealii-X CUDA BK1
// benchmarks (BwdTransHexKernel_QP_1D_SimpleMap): one work-item per
// quadrature point, explicit thread ids, a guard per step instead of a
// thread-stride loop.
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_SIMPLE_MAP_H
#define BK1_OFFLOAD_SIMPLE_MAP_H

#include <omp.h>

#include <cstddef>

#include "bk1_gpu.h"
#include "plane_tid.h"   // g_short_team

namespace bk1gpu {

// elements per team of the per-point kernels: fill about `threads`
// work-items, one per point of the nq^3 box
constexpr int point_batch(const int nq, const int threads)
{
    return (threads / (nq * nq * nq) > 1) ? threads / (nq * nq * nq) : 1;
}

// ---------------------------------------------------------------------------
// smap: a batch of elements per team, nq^3 work-item slots per element
// (work-item x -> element e = x / nq^3, slot l = x % nq^3), the team rounded
// up to whole sub-groups.  Every step is one guarded statement per slot, in
// the CUDA kernel's thread order (the last index of each step's box
// fastest), each output one dot product read from the two team boxes:
//
//   1 l < nm^3       copy in(e,i,j,k)                   -> w0(e,i,j,k)
//   2 l < nq nm^2    (p,j,k): sum_i w0(e,i,j,k) B(i,p)  -> w1(e,p,j,k)
//   3 l < nq^2 nm    (q,p,k): sum_j w1(e,p,j,k) B(j,q)  -> w0(e,q,p,k)
//   4 l < nq^3       (p,q,r): sum_k w0(e,q,p,k) B(k,r) * JxW(e,p,q,r) -> w1(e,p,q,r)
//   6 l < nq^2 nm    (p,q,k): sum_r w1(e,p,q,r) B(k,r)  -> w0(e,p,q,k)
//   7 l < nq nm^2    (p,j,k): sum_q w0(e,p,q,k) B(j,q)  -> w1(e,p,j,k)
//   8 l < nm^3       (i,j,k): sum_p w1(e,p,j,k) B(i,p)  -> out(e,i,j,k)
//
// Differences from the CUDA kernel: every step is guarded (the CUDA step 4
// is not, so its block must be exactly nq^3), a batch of elements per team
// instead of one per block with a grid stride, and step 8 writes out
// directly (the CUDA kernel stores to shared memory and copies, one barrier
// more).  The basis is copied to team memory once per team.  Each team must
// get `block` threads; the kernel sets `short_team` if it does not, and the
// harness reports it.  Every output is summed in the index order of
// SumFactorization.
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch>
void SumFactorizationSimpleMap(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    constexpr int block = round_up(batch * nq3, TEAM_ROUND);
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    static_assert(nm * nq <= block, "one work-item per basis entry for the copy");

    const std::size_t nteams = (nelmt + batch - 1) / batch;
    int short_team = 0;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3]) \
        map(tofrom: short_team)
    #pragma omp teams distribute thread_limit(block)
    for (std::size_t t = 0; t < nteams; ++t) {

        T Bs[nm * nq];
        T w0[batch * nq3];
        T w1[batch * nq3];

        const std::size_t e0 = t * batch;
        const int nb = (e0 + batch <= nelmt) ? batch : static_cast<int>(nelmt - e0);

        #pragma omp parallel num_threads(block)
        {
            const int x = omp_get_thread_num();
            if (x == 0 && omp_get_num_threads() < block) {
                #pragma omp atomic write
                short_team = 1;
            }
            if (x < nm * nq) {
                Bs[x] = basis[x];
            }
            const int e = x / nq3;
            const int l = x % nq3;
            const bool active = e < nb;
            const int eo = e * nq3;
            const std::size_t ge = e0 + static_cast<std::size_t>(e);

            // step 1 : copy
            if (active && l < nm3) {
                w0[eo + l] = in[ge * nm3 + l];
            }
            #pragma omp barrier

            // step 2 : direction 0
            if (active && l < nq * nm * nm) {
                const int p = l / (nm * nm);
                const int j = l % (nm * nm) / nm;
                const int k = l % nm;
                T tmp = 0;
                for (int i = 0; i < nm; ++i) {
                    tmp += w0[eo + i * nm * nm + j * nm + k] * Bs[i * nq + p];
                }
                w1[eo + l] = tmp;
            }
            #pragma omp barrier

            // step 3 : direction 1
            if (active && l < nq * nq * nm) {
                const int q = l / (nq * nm);
                const int p = l % (nq * nm) / nm;
                const int k = l % nm;
                T tmp = 0;
                for (int j = 0; j < nm; ++j) {
                    tmp += w1[eo + p * nm * nm + j * nm + k] * Bs[j * nq + q];
                }
                w0[eo + l] = tmp;
            }
            #pragma omp barrier

            // steps 4+5 : direction 2, times JxW
            if (active) {
                const int p = l / (nq * nq);
                const int q = l % (nq * nq) / nq;
                const int r = l % nq;
                T tmp = 0;
                for (int k = 0; k < nm; ++k) {
                    tmp += w0[eo + q * nq * nm + p * nm + k] * Bs[k * nq + r];
                }
                w1[eo + l] = tmp * JxW[ge * nq3 + l];
            }
            #pragma omp barrier

            // step 6 : direction 2 back
            if (active && l < nq * nq * nm) {
                const int p = l / (nq * nm);
                const int q = l % (nq * nm) / nm;
                const int k = l % nm;
                T tmp = 0;
                for (int r = 0; r < nq; ++r) {
                    tmp += w1[eo + p * nq * nq + q * nq + r] * Bs[k * nq + r];
                }
                w0[eo + l] = tmp;
            }
            #pragma omp barrier

            // step 7 : direction 1 back
            if (active && l < nq * nm * nm) {
                const int p = l / (nm * nm);
                const int j = l % (nm * nm) / nm;
                const int k = l % nm;
                T tmp = 0;
                for (int q = 0; q < nq; ++q) {
                    tmp += w0[eo + p * nq * nm + q * nm + k] * Bs[j * nq + q];
                }
                w1[eo + l] = tmp;
            }
            #pragma omp barrier

            // step 8 : direction 0 back, straight to out
            if (active && l < nm3) {
                const int i = l / (nm * nm);
                const int j = l % (nm * nm) / nm;
                const int k = l % nm;
                T tmp = 0;
                for (int p = 0; p < nq; ++p) {
                    tmp += w1[eo + p * nm * nm + j * nm + k] * Bs[i * nq + p];
                }
                out[ge * nm3 + l] = tmp;
            }
        }
    }
    g_short_team |= short_team;
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_SIMPLE_MAP_H
