// plane_tid.h -- plane kernels with explicit thread ids and barriers
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_PLANE_TID_H
#define BK1_OFFLOAD_PLANE_TID_H

#include <omp.h>

#include <cstddef>

#include "bk_common.h"
#include "bk1_sumfact.h"
#include "bk1_gpu.h"

namespace bk1gpu {

using bk::ndview;

// ---------------------------------------------------------------------------
// planeT / planeX: the plane kernel with explicit thread ids instead of
// worksharing loops, so a thread keeps its column in registers across a
// barrier.  planeT keeps the two boxes (4 barriers); planeX works in one
// box (7 barriers: read into registers, barrier, write back), which halves
// the team-shared memory and so doubles the teams that fit in an Xe-core's
// 64 KB of SLM.  Each team must get team_size threads (one per work item of
// the widest phase); the kernel sets `short_team` if it does not, and the
// host asserts on it after the launch.
// ---------------------------------------------------------------------------
inline int g_short_team = 0;

template <typename T, int nq, int batch, bool one_box, typename index_t = int>
void SumFactorizationPlaneTid(
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
    constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");
    constexpr int nbox = one_box ? 1 : 2;

    const std::size_t nteams = (nelmt + batch - 1) / batch;
    int short_team = 0;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3]) \
        map(tofrom: short_team)
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        T Bs[nm * nq];
        T box[nbox * batch * nq3];
        const ndview<const T, nm, nq> B{Bs};
        const ndview<T, batch, nq, nq, nq> s0{box};
        const ndview<T, batch, nq, nq, nq> s1{box + (nbox - 1) * batch * nq3};

        const std::size_t e0 = t * batch;
        const index_t nb = (e0 + batch <= nelmt) ? batch : static_cast<index_t>(nelmt - e0);

        #pragma omp parallel
        {
            const index_t tid = omp_get_thread_num();
            if (tid == 0 && omp_get_num_threads() < team_size) {
                #pragma omp atomic write
                short_team = 1;
            }
            if (tid < nm * nq) {
                Bs[tid] = basis[tid];
            }
            // thread -> (b, x, y) of the nq x nq plane; phases A and E use
            // x, y < nm, phases B and D x < nq, y < nm, phase C both < nq
            const index_t b = tid / (nq * nq);
            const index_t x = (tid / nq) % nq;
            const index_t y = tid % nq;
            const bool in_batch = b < nb;
            T v[nq];
            #pragma omp barrier

            // phase A : thread (b,j,k) = (b,x,y), contract direction 0
            if (in_batch && x < nm && y < nm) {
                const ndview<const T, nm, nm, nm> e_in{in + (e0 + b) * nm3};
                T u[nm];
                UNROLL
                for (index_t i = 0; i < nm; ++i) {
                    u[i] = e_in(i, x, y);
                }
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] = 0;
                }
                UNROLL
                for (index_t i = 0; i < nm; ++i) {
                    UNROLL
                    for (index_t p = 0; p < nq; ++p) {
                        v[p] += u[i] * B(i, p);
                    }
                }
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    s0(b, p, x, y) = v[p];
                }
            }
            #pragma omp barrier

            // phase B : thread (b,q,k) = (b,x,y), contract direction 1
            const bool act_b = in_batch && y < nm;
            if (act_b) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] = 0;
                }
                UNROLL
                for (index_t j = 0; j < nm; ++j) {
                    const T bj = B(j, x);
                    UNROLL
                    for (index_t p = 0; p < nq; ++p) {
                        v[p] += s0(b, p, j, y) * bj;
                    }
                }
            }
            if constexpr (one_box) {
                #pragma omp barrier
            }
            if (act_b) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    s1(b, p, x, y) = v[p];
                }
            }
            #pragma omp barrier

            // phase C : thread (b,q,r) = (b,x,y), contract direction 2, * JxW
            if (in_batch) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] = 0;
                }
                UNROLL
                for (index_t k = 0; k < nm; ++k) {
                    const T bk = B(k, y);
                    UNROLL
                    for (index_t p = 0; p < nq; ++p) {
                        v[p] += s1(b, p, x, k) * bk;
                    }
                }
                const ndview<const T, nq, nq, nq> e_JxW{JxW + (e0 + b) * nq3};
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] *= e_JxW(p, x, y);
                }
            }
            if constexpr (one_box) {
                #pragma omp barrier
            }
            if (in_batch) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    s0(b, p, x, y) = v[p];
                }
            }
            #pragma omp barrier

            // phase D : thread (b,q,k) = (b,x,y), contract direction 2 back
            if (act_b) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] = 0;
                }
                UNROLL
                for (index_t r = 0; r < nq; ++r) {
                    const T bk = B(y, r);
                    UNROLL
                    for (index_t p = 0; p < nq; ++p) {
                        v[p] += s0(b, p, x, r) * bk;
                    }
                }
            }
            if constexpr (one_box) {
                #pragma omp barrier
            }
            if (act_b) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    s1(b, p, x, y) = v[p];
                }
            }
            #pragma omp barrier

            // phase E : thread (b,j,k) = (b,x,y), contract direction 1 back,
            // then direction 0 back in registers, store out(:, j, k)
            if (in_batch && x < nm && y < nm) {
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    v[p] = 0;
                }
                UNROLL
                for (index_t q = 0; q < nq; ++q) {
                    const T bj = B(x, q);
                    UNROLL
                    for (index_t p = 0; p < nq; ++p) {
                        v[p] += s1(b, p, q, y) * bj;
                    }
                }
                T w[nm];
                UNROLL
                for (index_t i = 0; i < nm; ++i) {
                    w[i] = 0;
                }
                UNROLL
                for (index_t p = 0; p < nq; ++p) {
                    UNROLL
                    for (index_t i = 0; i < nm; ++i) {
                        w[i] += v[p] * B(i, p);
                    }
                }
                const ndview<T, nm, nm, nm> e_out{out + (e0 + b) * nm3};
                UNROLL
                for (index_t i = 0; i < nm; ++i) {
                    e_out(i, x, y) = w[i];
                }
            }
        }
    }
    g_short_team |= short_team;
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_PLANE_TID_H
