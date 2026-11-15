// teams.h -- one element per team, a parallel for collapse(3) per step
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_TEAMS_H
#define BK1_OFFLOAD_TEAMS_H

#include <omp.h>

#include <cstddef>

#include "bk_common.h"
#include "bk1_sumfact.h"
#include "bk1_gpu.h"

namespace bk1gpu {

using bk::ndview;

// ---------------------------------------------------------------------------
// teams: one element per team, the work arrays at team scope, a
// `parallel for collapse(3)` per step (the end of each region is the barrier)
// ---------------------------------------------------------------------------
template <typename T, int nq, typename index_t = int>
void SumFactorizationTeams(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = round_up(nq3, TEAM_ROUND);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t e = 0; e < nelmt; ++e) {

        T wsp0[nq3];
        T wsp1[nq3];
        const ndview<const T, nm, nq> B{basis};
        const ndview<T, nq, nq, nq> w0{wsp0};
        const ndview<T, nq, nq, nq> w1{wsp1};
        const ndview<const T, nm, nm, nm> e_in{in + e * nm3};
        const ndview<T, nm, nm, nm> e_out{out + e * nm3};
        const ndview<const T, nq, nq, nq> e_JxW{JxW + e * nq3};

        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    w0(i, j, k) = e_in(i, j, k);
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; ++i) {
                        tmp += w0(i, j, k) * B(i, p);
                    }
                    w1(p, j, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; ++j) {
                        tmp += w1(p, j, k) * B(j, q);
                    }
                    w0(q, p, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; ++k) {
                        tmp += w0(q, p, k) * B(k, r);
                    }
                    w1(p, q, r) = tmp * e_JxW(p, q, r);
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r) {
                        tmp += w1(p, q, r) * B(k, r);
                    }
                    w0(q, p, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q) {
                        tmp += w0(q, p, k) * B(j, q);
                    }
                    w1(p, j, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p) {
                        tmp += w1(p, j, k) * B(i, p);
                    }
                    e_out(i, j, k) = tmp;
                }
            }
        }
    }
}

// teamsD: teams with step 2 reading in directly (no copy step)
template <typename T, int nq, typename index_t = int>
void SumFactorizationTeamsD(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = round_up(nq3, TEAM_ROUND);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t e = 0; e < nelmt; ++e) {

        T wsp0[nq3];
        T wsp1[nq3];
        const ndview<const T, nm, nq> B{basis};
        const ndview<T, nq, nq, nq> w0{wsp0};
        const ndview<T, nq, nq, nq> w1{wsp1};
        const ndview<const T, nm, nm, nm> e_in{in + e * nm3};
        const ndview<T, nm, nm, nm> e_out{out + e * nm3};
        const ndview<const T, nq, nq, nq> e_JxW{JxW + e * nq3};

        #pragma omp parallel for collapse(3)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; ++i) {
                        tmp += e_in(i, j, k) * B(i, p);
                    }
                    w1(p, j, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; ++j) {
                        tmp += w1(p, j, k) * B(j, q);
                    }
                    w0(q, p, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; ++k) {
                        tmp += w0(q, p, k) * B(k, r);
                    }
                    w1(p, q, r) = tmp * e_JxW(p, q, r);
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r) {
                        tmp += w1(p, q, r) * B(k, r);
                    }
                    w0(q, p, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q) {
                        tmp += w0(q, p, k) * B(j, q);
                    }
                    w1(p, j, k) = tmp;
                }
            }
        }
        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p) {
                        tmp += w1(p, j, k) * B(i, p);
                    }
                    e_out(i, j, k) = tmp;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// teams1p: as teams, but one parallel region per element with an
// `omp for collapse(3)` per step (implicit barriers)
// ---------------------------------------------------------------------------
template <typename T, int nq, typename index_t = int>
void SumFactorizationTeams1P(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    [[maybe_unused]] constexpr int team_size = round_up(nq3, TEAM_ROUND);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t e = 0; e < nelmt; ++e) {

        T wsp0[nq3];
        T wsp1[nq3];
        const ndview<const T, nm, nq> B{basis};
        const ndview<T, nq, nq, nq> w0{wsp0};
        const ndview<T, nq, nq, nq> w1{wsp1};
        const ndview<const T, nm, nm, nm> e_in{in + e * nm3};
        const ndview<T, nm, nm, nm> e_out{out + e * nm3};
        const ndview<const T, nq, nq, nq> e_JxW{JxW + e * nq3};

        #pragma omp parallel
        {
            #pragma omp for collapse(3)
            for (index_t i = 0; i < nm; ++i) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        w0(i, j, k) = e_in(i, j, k);
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    for (index_t j = 0; j < nm; ++j) {
                        T tmp = 0;
                        for (index_t i = 0; i < nm; ++i) {
                            tmp += w0(i, j, k) * B(i, p);
                        }
                        w1(p, j, k) = tmp;
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    for (index_t k = 0; k < nm; ++k) {
                        T tmp = 0;
                        for (index_t j = 0; j < nm; ++j) {
                            tmp += w1(p, j, k) * B(j, q);
                        }
                        w0(q, p, k) = tmp;
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t r = 0; r < nq; ++r) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t p = 0; p < nq; ++p) {
                        T tmp = 0;
                        for (index_t k = 0; k < nm; ++k) {
                            tmp += w0(q, p, k) * B(k, r);
                        }
                        w1(p, q, r) = tmp * e_JxW(p, q, r);
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t k = 0; k < nm; ++k) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t p = 0; p < nq; ++p) {
                        T tmp = 0;
                        for (index_t r = 0; r < nq; ++r) {
                            tmp += w1(p, q, r) * B(k, r);
                        }
                        w0(q, p, k) = tmp;
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    for (index_t p = 0; p < nq; ++p) {
                        T tmp = 0;
                        for (index_t q = 0; q < nq; ++q) {
                            tmp += w0(q, p, k) * B(j, q);
                        }
                        w1(p, j, k) = tmp;
                    }
                }
            }
            #pragma omp for collapse(3)
            for (index_t i = 0; i < nm; ++i) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        T tmp = 0;
                        for (index_t p = 0; p < nq; ++p) {
                            tmp += w1(p, j, k) * B(i, p);
                        }
                        e_out(i, j, k) = tmp;
                    }
                }
            }
        }
    }
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_TEAMS_H
