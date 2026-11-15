// plane.h -- plane kernels: nq x nq threads per element, register columns
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_PLANE_H
#define BK1_OFFLOAD_PLANE_H

#include <omp.h>

#include <cstddef>

#include "bk_common.h"
#include "bk1_sumfact.h"
#include "bk1_gpu.h"

namespace bk1gpu {

using bk::ndview;

// ---------------------------------------------------------------------------
// plane: the float version of BK1_ff's SumFactorizationFFPlane -- a batch of
// elements per team, a plane of nq x nq threads per element, each thread
// owning the column through its point along the first index in registers;
// five phases with a barrier between them, the contraction index as the
// outer loop so that each shared value is loaded once per column.  The
// basis is copied to team-shared memory first.  Every output is summed in
// the same order as SumFactorization.
//
//   A: thread (b,j,k): in(i,j,k)    -> contract i -> s[p]       -> sm0(b,p,j,k)
//   B: thread (b,q,k): sm0(b,p,j,k) -> contract j -> t[p]       -> sm1(b,p,q,k)
//   C: thread (b,q,r): sm1(b,p,q,k) -> contract k -> v[p] * JxW -> sm0(b,p,q,r)
//   D: thread (b,q,k): sm0(b,p,q,r) -> contract r -> w[p]       -> sm1(b,p,q,k)
//   E: thread (b,j,k): sm1(b,p,q,k) -> contract q -> x[p], contract p -> out(i,j,k)
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, typename index_t = int>
void SumFactorizationPlane(
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
    [[maybe_unused]] constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");

    const std::size_t nteams = (nelmt + batch - 1) / batch;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the basis and two nq^3 boxes per element.
        T Bs[nm * nq];
        T sm0[batch * nq3];
        T sm1[batch * nq3];
        const ndview<const T, nm, nq> B{Bs};
        const ndview<T, batch, nq, nq, nq> s0{sm0};
        const ndview<T, batch, nq, nq, nq> s1{sm1};

        const std::size_t e0 = t * batch;
        const index_t nb = (e0 + batch <= nelmt) ? batch : static_cast<index_t>(nelmt - e0);

        #pragma omp parallel
        {
            #pragma omp for
            for (index_t i = 0; i < nm * nq; ++i) {
                Bs[i] = basis[i];
            }

            // phase A : load the column in(:, j, k), contract direction 0
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<const T, nm, nm, nm> e_in{in + (e0 + b) * nm3};
                        T u[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            u[i] = e_in(i, j, k);
                        }
                        T s[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s[p] = 0;
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                s[p] += u[i] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, p, j, k) = s[p];
                        }
                    }
                }
            }

            // phase B : contract direction 1
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t j = 0; j < nm; ++j) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, p, j, k) * bj;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, p, q, k) = v[p];
                        }
                    }
                }
            }

            // phase C : contract direction 2, multiply with JxW
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        const ndview<const T, nq, nq, nq> e_JxW{JxW + (e0 + b) * nq3};
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t k = 0; k < nm; ++k) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s1(b, p, q, k) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, p, q, r) = v[p] * e_JxW(p, q, r);
                        }
                    }
                }
            }

            // phase D : contract direction 2 back
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t r = 0; r < nq; ++r) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, p, q, r) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, p, q, k) = v[p];
                        }
                    }
                }
            }

            // phase E : contract direction 1 back, then direction 0 back in
            // registers, store the column out(:, j, k)
            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<T, nm, nm, nm> e_out{out + (e0 + b) * nm3};
                        T x[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            x[p] = 0;
                        }
                        UNROLL
                        for (index_t q = 0; q < nq; ++q) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                x[p] += s1(b, p, q, k) * bj;
                            }
                        }
                        T y[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            y[i] = 0;
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            UNROLL
                            for (index_t i = 0; i < nm; ++i) {
                                y[i] += x[p] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            e_out(i, j, k) = y[i];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// planeC: the plane kernel with a compile-time batch in every team and
// 32-bit offsets inside the team.  nelmt / batch full teams run in one
// launch, the remaining nelmt % batch elements in a second launch with
// batch 1.  (VTune: the plane kernel's integer pipe runs 2.5x the float
// instructions, much of it index decoding with run-time bounds.)
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, typename index_t = int>
void PlaneFull(
    const std::size_t nteams,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nteams*batch*nq3], in[:nteams*batch*nm3]) \
        map(from: out[:nteams*batch*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the basis and two nq^3 boxes per element.
        T Bs[nm * nq];
        T sm0[batch * nq3];
        T sm1[batch * nq3];
        const ndview<const T, nm, nq> B{Bs};
        const ndview<T, batch, nq, nq, nq> s0{sm0};
        const ndview<T, batch, nq, nq, nq> s1{sm1};

        // Every team is full: the batch is a compile-time constant, and the
        // team's base pointers are formed once, so the per-element offsets
        // inside the phases are 32-bit.
        constexpr index_t nb = batch;
        const T* t_in  = in  + t * std::size_t(batch * nm3);
        const T* t_JxW = JxW + t * std::size_t(batch * nq3);
        T*       t_out = out + t * std::size_t(batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (index_t i = 0; i < nm * nq; ++i) {
                Bs[i] = basis[i];
            }

            // phase A : load the column in(:, j, k), contract direction 0
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<const T, nm, nm, nm> e_in{t_in + b * nm3};
                        T u[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            u[i] = e_in(i, j, k);
                        }
                        T s[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s[p] = 0;
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                s[p] += u[i] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, p, j, k) = s[p];
                        }
                    }
                }
            }

            // phase B : contract direction 1
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t j = 0; j < nm; ++j) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, p, j, k) * bj;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, p, q, k) = v[p];
                        }
                    }
                }
            }

            // phase C : contract direction 2, multiply with JxW
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        const ndview<const T, nq, nq, nq> e_JxW{t_JxW + b * nq3};
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t k = 0; k < nm; ++k) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s1(b, p, q, k) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, p, q, r) = v[p] * e_JxW(p, q, r);
                        }
                    }
                }
            }

            // phase D : contract direction 2 back
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t r = 0; r < nq; ++r) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, p, q, r) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, p, q, k) = v[p];
                        }
                    }
                }
            }

            // phase E : contract direction 1 back, then direction 0 back in
            // registers, store the column out(:, j, k)
            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<T, nm, nm, nm> e_out{t_out + b * nm3};
                        T x[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            x[p] = 0;
                        }
                        UNROLL
                        for (index_t q = 0; q < nq; ++q) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                x[p] += s1(b, p, q, k) * bj;
                            }
                        }
                        T y[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            y[i] = 0;
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            UNROLL
                            for (index_t i = 0; i < nm; ++i) {
                                y[i] += x[p] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            e_out(i, j, k) = y[i];
                        }
                    }
                }
            }
        }
    }
}


template <typename T, int nq, int batch, typename index_t = int>
void SumFactorizationPlaneC(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nq3 = nq * nq * nq;
    constexpr std::size_t nm3 = nm * nm * nm;
    const std::size_t nfull = nelmt / batch;
    const std::size_t rest = nelmt - nfull * batch;
    if (nfull > 0) {
        PlaneFull<T, nq, batch, index_t>(nfull, basis, JxW, in, out);
    }
    if (rest > 0) {
        const std::size_t off = nfull * batch;
        PlaneFull<T, nq, 1, index_t>(rest, basis, JxW + off * nq3, in + off * nm3, out + off * nm3);
    }
}

// ---------------------------------------------------------------------------
// planeV: planeC with the column index innermost in the team-shared boxes,
// s(b, x, y, p) instead of s(b, p, x, y), so every column a thread reads or
// writes is contiguous: one vector SLM load/store of nq floats instead of nq
// scalar ones, each with its own address computation (DG2's LSC has no
// immediate offsets).
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, int pad = 0, typename index_t = int>
void PlaneFullV(
    const std::size_t nteams,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nteams*batch*nq3], in[:nteams*batch*nm3]) \
        map(from: out[:nteams*batch*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the basis and two nq^3 boxes per element.
        T Bs[nm * nq];
        // the column (last extent) padded by `pad` to spread lanes over banks
        T sm0[batch * nq * nq * (nq + pad)];
        T sm1[batch * nq * nq * (nq + pad)];
        const ndview<const T, nm, nq> B{Bs};
        const ndview<T, batch, nq, nq, nq + pad> s0{sm0};
        const ndview<T, batch, nq, nq, nq + pad> s1{sm1};

        // Every team is full: the batch is a compile-time constant, and the
        // team's base pointers are formed once, so the per-element offsets
        // inside the phases are 32-bit.
        constexpr index_t nb = batch;
        const T* t_in  = in  + t * std::size_t(batch * nm3);
        const T* t_JxW = JxW + t * std::size_t(batch * nq3);
        T*       t_out = out + t * std::size_t(batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (index_t i = 0; i < nm * nq; ++i) {
                Bs[i] = basis[i];
            }

            // phase A : load the column in(:, j, k), contract direction 0
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<const T, nm, nm, nm> e_in{t_in + b * nm3};
                        T u[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            u[i] = e_in(i, j, k);
                        }
                        T s[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s[p] = 0;
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                s[p] += u[i] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, j, k, p) = s[p];
                        }
                    }
                }
            }

            // phase B : contract direction 1
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t j = 0; j < nm; ++j) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, j, k, p) * bj;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase C : contract direction 2, multiply with JxW
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        const ndview<const T, nq, nq, nq> e_JxW{t_JxW + b * nq3};
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t k = 0; k < nm; ++k) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s1(b, q, k, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, q, r, p) = v[p] * e_JxW(p, q, r);
                        }
                    }
                }
            }

            // phase D : contract direction 2 back
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t r = 0; r < nq; ++r) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, q, r, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase E : contract direction 1 back, then direction 0 back in
            // registers, store the column out(:, j, k)
            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<T, nm, nm, nm> e_out{t_out + b * nm3};
                        T x[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            x[p] = 0;
                        }
                        UNROLL
                        for (index_t q = 0; q < nq; ++q) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                x[p] += s1(b, q, k, p) * bj;
                            }
                        }
                        T y[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            y[i] = 0;
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            UNROLL
                            for (index_t i = 0; i < nm; ++i) {
                                y[i] += x[p] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            e_out(i, j, k) = y[i];
                        }
                    }
                }
            }
        }
    }
}


template <typename T, int nq, int batch, int pad = 0, typename index_t = int>
void SumFactorizationPlaneV(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nq3 = nq * nq * nq;
    constexpr std::size_t nm3 = nm * nm * nm;
    const std::size_t nfull = nelmt / batch;
    const std::size_t rest = nelmt - nfull * batch;
    if (nfull > 0) {
        PlaneFullV<T, nq, batch, pad, index_t>(nfull, basis, JxW, in, out);
    }
    if (rest > 0) {
        const std::size_t off = nfull * batch;
        PlaneFullV<T, nq, 1, pad, index_t>(rest, basis, JxW + off * nq3, in + off * nm3, out + off * nm3);
    }
}

// ---------------------------------------------------------------------------
// planeG: planeV without the SLM copy of the basis (A/B test): the phases
// read the basis from global memory and rely on the caches, and the copy
// loop's barrier goes with it.
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, int pad = 0, typename index_t = int>
void PlaneFullG(
    const std::size_t nteams,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nteams*batch*nq3], in[:nteams*batch*nm3]) \
        map(from: out[:nteams*batch*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: two nq^3 boxes per element; the basis is read from
        // global memory (through the caches), not copied to SLM.
        // the column (last extent) padded by `pad` to spread lanes over banks
        T sm0[batch * nq * nq * (nq + pad)];
        T sm1[batch * nq * nq * (nq + pad)];
        const ndview<const T, nm, nq> B{basis};
        const ndview<T, batch, nq, nq, nq + pad> s0{sm0};
        const ndview<T, batch, nq, nq, nq + pad> s1{sm1};

        // Every team is full: the batch is a compile-time constant, and the
        // team's base pointers are formed once, so the per-element offsets
        // inside the phases are 32-bit.
        constexpr index_t nb = batch;
        const T* t_in  = in  + t * std::size_t(batch * nm3);
        const T* t_JxW = JxW + t * std::size_t(batch * nq3);
        T*       t_out = out + t * std::size_t(batch * nm3);

        #pragma omp parallel
        {

            // phase A : load the column in(:, j, k), contract direction 0
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<const T, nm, nm, nm> e_in{t_in + b * nm3};
                        T u[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            u[i] = e_in(i, j, k);
                        }
                        T s[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s[p] = 0;
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                s[p] += u[i] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, j, k, p) = s[p];
                        }
                    }
                }
            }

            // phase B : contract direction 1
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t j = 0; j < nm; ++j) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, j, k, p) * bj;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase C : contract direction 2, multiply with JxW
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        const ndview<const T, nq, nq, nq> e_JxW{t_JxW + b * nq3};
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t k = 0; k < nm; ++k) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s1(b, q, k, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, q, r, p) = v[p] * e_JxW(p, q, r);
                        }
                    }
                }
            }

            // phase D : contract direction 2 back
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t r = 0; r < nq; ++r) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, q, r, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase E : contract direction 1 back, then direction 0 back in
            // registers, store the column out(:, j, k)
            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        const ndview<T, nm, nm, nm> e_out{t_out + b * nm3};
                        T x[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            x[p] = 0;
                        }
                        UNROLL
                        for (index_t q = 0; q < nq; ++q) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                x[p] += s1(b, q, k, p) * bj;
                            }
                        }
                        T y[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            y[i] = 0;
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            UNROLL
                            for (index_t i = 0; i < nm; ++i) {
                                y[i] += x[p] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            e_out(i, j, k) = y[i];
                        }
                    }
                }
            }
        }
    }
}


template <typename T, int nq, int batch, int pad = 0, typename index_t = int>
void SumFactorizationPlaneG(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nq3 = nq * nq * nq;
    constexpr std::size_t nm3 = nm * nm * nm;
    const std::size_t nfull = nelmt / batch;
    const std::size_t rest = nelmt - nfull * batch;
    if (nfull > 0) {
        PlaneFullG<T, nq, batch, pad, index_t>(nfull, basis, JxW, in, out);
    }
    if (rest > 0) {
        const std::size_t off = nfull * batch;
        PlaneFullG<T, nq, 1, pad, index_t>(rest, basis, JxW + off * nq3, in + off * nm3, out + off * nm3);
    }
}

// ---------------------------------------------------------------------------
// planeL: planeV with small [=] lambdas as accessors instead of ndview
// (pointer arguments carry a _p suffix, the accessors are named without it)
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, int pad = 0, typename index_t = int>
void PlaneFullL(
    const std::size_t nteams,
    const T* __restrict__ basis_p,
    const T* __restrict__ JxW_p,
    const T* __restrict__ in_p,
    T* __restrict__ out_p)
{
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = batch * nq * nq;
    static_assert(team_size <= 1024, "a work-group holds at most 1024 work-items");

    #pragma omp target \
        map(to: basis_p[:nm*nq]) \
        map(to: JxW_p[:nteams*batch*nq3], in_p[:nteams*batch*nm3]) \
        map(from: out_p[:nteams*batch*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the basis and two nq^3 boxes per element.
        T Bs[nm * nq];
        // the column (last extent) padded by `pad` to spread lanes over banks
        T sm0[batch * nq * nq * (nq + pad)];
        T sm1[batch * nq * nq * (nq + pad)];
        T* const Bs_p = Bs;
        T* const sm0_p = sm0;
        T* const sm1_p = sm1;
        constexpr int nc = nq + pad;
        const auto B  = [=](int i, int p) -> T& { return Bs_p[i * nq + p]; };
        const auto s0 = [=](int b, int x, int y, int p) -> T& { return sm0_p[((b * nq + x) * nq + y) * nc + p]; };
        const auto s1 = [=](int b, int x, int y, int p) -> T& { return sm1_p[((b * nq + x) * nq + y) * nc + p]; };

        // Every team is full: the batch is a compile-time constant, and the
        // team's base pointers are formed once, so the per-element offsets
        // inside the phases are 32-bit.
        constexpr index_t nb = batch;
        const T* t_in  = in_p  + t * std::size_t(batch * nm3);
        const T* t_JxW = JxW_p + t * std::size_t(batch * nq3);
        T*       t_out = out_p + t * std::size_t(batch * nm3);
        const auto in  = [=](int b, int i, int j, int k) -> T  { return t_in [((b * nm + i) * nm + j) * nm + k]; };
        const auto JxW = [=](int b, int p, int q, int r) -> T  { return t_JxW[((b * nq + p) * nq + q) * nq + r]; };
        const auto out = [=](int b, int i, int j, int k) -> T& { return t_out[((b * nm + i) * nm + j) * nm + k]; };

        #pragma omp parallel
        {
            #pragma omp for
            for (index_t i = 0; i < nm * nq; ++i) {
                Bs[i] = basis_p[i];
            }

            // phase A : load the column in(:, j, k), contract direction 0
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        T u[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            u[i] = in(b, i, j, k);
                        }
                        T s[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s[p] = 0;
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                s[p] += u[i] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, j, k, p) = s[p];
                        }
                    }
                }
            }

            // phase B : contract direction 1
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t j = 0; j < nm; ++j) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, j, k, p) * bj;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase C : contract direction 2, multiply with JxW
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t k = 0; k < nm; ++k) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s1(b, q, k, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s0(b, q, r, p) = v[p] * JxW(b, p, q, r);
                        }
                    }
                }
            }

            // phase D : contract direction 2 back
            #pragma omp for collapse(3)
            for (index_t b = 0; b < nb; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        T v[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            v[p] = 0;
                        }
                        UNROLL
                        for (index_t r = 0; r < nq; ++r) {
                            const T bk = B(k, r);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                v[p] += s0(b, q, r, p) * bk;
                            }
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            s1(b, q, k, p) = v[p];
                        }
                    }
                }
            }

            // phase E : contract direction 1 back, then direction 0 back in
            // registers, store the column out(:, j, k)
            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < nb; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        T x[nq];
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            x[p] = 0;
                        }
                        UNROLL
                        for (index_t q = 0; q < nq; ++q) {
                            const T bj = B(j, q);
                            UNROLL
                            for (index_t p = 0; p < nq; ++p) {
                                x[p] += s1(b, q, k, p) * bj;
                            }
                        }
                        T y[nm];
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            y[i] = 0;
                        }
                        UNROLL
                        for (index_t p = 0; p < nq; ++p) {
                            UNROLL
                            for (index_t i = 0; i < nm; ++i) {
                                y[i] += x[p] * B(i, p);
                            }
                        }
                        UNROLL
                        for (index_t i = 0; i < nm; ++i) {
                            out(b, i, j, k) = y[i];
                        }
                    }
                }
            }
        }
    }
}


template <typename T, int nq, int batch, typename index_t = int>
void SumFactorizationPlaneL(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nq3 = nq * nq * nq;
    constexpr std::size_t nm3 = nm * nm * nm;
    const std::size_t nfull = nelmt / batch;
    const std::size_t rest = nelmt - nfull * batch;
    if (nfull > 0) {
        PlaneFullL<T, nq, batch, 0, index_t>(nfull, basis, JxW, in, out);
    }
    if (rest > 0) {
        const std::size_t off = nfull * batch;
        PlaneFullL<T, nq, 1, 0, index_t>(rest, basis, JxW + off * nq3, in + off * nm3, out + off * nm3);
    }
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_PLANE_H
