#ifndef BK1_SUMFACT_H
#define BK1_SUMFACT_H

// bk1_sumfact.h -- the BK1 (mass operator) sum factorization, generic in the
// arithmetic type T.  T is float or double for the plain kernels and ffloat
// (float_float.h) for the float-float kernel: T{} must be zero, T needs
// `+=`, `*` and `*=`, and it must be trivially copyable so that the OpenMP
// maps can move the arrays bitwise.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "bk_common.h"

namespace bk {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    static_assert(std::is_trivially_copyable_v<T>, "T is mapped bitwise to the device");
    constexpr int nm = nq - 1;

    // Plain index macros instead of accessor objects: the offload compilers
    // then see nothing but arrays of T in the loop body.
    // B(mode i, quad point p), shared by all directions.
    #define B(i, p)          basis[(i) * nq + (p)]
    // The two nq^3 work boxes of one element; steps address sub-slices of
    // each box, and every step assigns its full output sub-slice, so no
    // zeroing is needed.
    #define wsp0(i, j, k)    wsp0[nq * nq * (i) + nq * (j) + (k)]
    #define wsp1(i, j, k)    wsp1[nq * nq * (i) + nq * (j) + (k)]
    // The fields of element e.
    #define e_in(i, j, k)    in [nm * nm * nm * e + nm * nm * (i) + nm * (j) + (k)]
    #define e_out(i, j, k)   out[nm * nm * nm * e + nm * nm * (i) + nm * (j) + (k)]
    #define e_JxW(i, j, k)   JxW[nq * nq * nq * e + nq * nq * (i) + nq * (j) + (k)]

    // Elements over teams; the inner `loop` constructs let the compiler use a
    // team's threads inside an element where it maps elements to teams.
    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq*nq*nq], in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop bind(teams)
    for (std::size_t e = 0; e < nelmt; ++e) {

        T wsp0[nq * nq * nq];
        T wsp1[nq * nq * nq];

        // step-1 : copy in -> wsp0 (nm^3 sub-block of the nq^3 box)
        #pragma omp loop collapse(3) bind(thread)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    wsp0(i, j, k) = e_in(i, j, k);
                }
            }
        }

        // step-2 : direction 0
        #pragma omp loop collapse(3) bind(thread)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    T tmp{};
                    for (index_t i = 0; i < nm; ++i) {
                        tmp += wsp0(i, j, k) * B(i, p);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }

        // step-3 : direction 1
        #pragma omp loop collapse(3) bind(thread)
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp{};
                    for (index_t j = 0; j < nm; ++j) {
                        tmp += wsp1(p, j, k) * B(j, q);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }

        // step-4 : direction 2
        #pragma omp loop collapse(3) bind(thread)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp{};
                    for (index_t k = 0; k < nm; ++k) {
                        tmp += wsp0(q, p, k) * B(k, r);
                    }
                    wsp1(p, q, r) = tmp;
                }
            }
        }

        // Reverse operations

        // step-5 : multiply with weights and determinant of Jacobi
        #pragma omp loop collapse(3) bind(thread)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    wsp1(p, q, r) *= e_JxW(p, q, r);
                }
            }
        }

        // step-6 : direction 2
        #pragma omp loop collapse(3) bind(thread)
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp{};
                    for (index_t r = 0; r < nq; ++r) {
                        tmp += wsp1(p, q, r) * B(k, r);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }

        // step-7 : direction 1
        #pragma omp loop collapse(3) bind(thread)
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp{};
                    for (index_t q = 0; q < nq; ++q) {
                        tmp += wsp0(q, p, k) * B(j, q);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }

        // step-8 : direction 0
        #pragma omp loop collapse(3) bind(thread)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp{};
                    for (index_t p = 0; p < nq; ++p) {
                        tmp += wsp1(p, j, k) * B(i, p);
                    }
                    wsp0(i, j, k) = tmp;
                }
            }
        }

        // step-9 : copy wsp0 -> out
        #pragma omp loop collapse(3) bind(thread)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out(i, j, k) = wsp0(i, j, k);
                }
            }
        }
    }

    #undef B
    #undef wsp0
    #undef wsp1
    #undef e_in
    #undef e_out
    #undef e_JxW
}

// Benchmark harness shared by the drivers: keep the arrays mapped on the
// device across ntests runs of the kernel and return the minimum wall time
// of one run in seconds.
template <typename T, int nq>
double time_sumfact(
    const std::size_t nelmt, const int ntests,
    const T* basis, const T* JxW, const T* in, T* out)
{
    constexpr int nm = nq - 1;
    // the extents only appear in the map clauses
    [[maybe_unused]] const std::size_t size_inout = nelmt * nm * nm * nm;
    [[maybe_unused]] const std::size_t size_JxW   = nelmt * nq * nq * nq;
    [[maybe_unused]] constexpr std::size_t size_basis = nm * nq;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma omp target data \
        map(to: basis[:size_basis]) \
        map(to: JxW[:size_JxW], in[:size_inout]) \
        map(tofrom: out[:size_inout])
    for (int t = 0; t < ntests; ++t) {
        const auto start = high_resolution_clock::now();
        SumFactorization<T, nq>(nelmt, basis, JxW, in, out);
        const auto stop = high_resolution_clock::now();
        const duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }
    return elapsed;
}

} // namespace bk

#endif // BK1_SUMFACT_H
