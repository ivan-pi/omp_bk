#ifndef BK1_KERNEL_H
#define BK1_KERNEL_H

#include <cstddef>

#include "bk_common.h"

namespace bk::bk1 {

// BK1: mass-matrix operator via sum factorization.
//   nq = number of quadrature points (per direction), nm = nq - 1 modes.
// The kernel body is header-only so that it can be instantiated both by the
// timing driver (BK1.cpp) and by the arithmetic-intensity tool, which counts
// flops by substituting an operation-counting scalar type for T.
template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;

    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;
    using nq_cview = ndview<const T, nq, nq, nq>;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq*nq*nq], in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
        const ndview<const T, nm, nq> B{basis};

    	// Work arrays: two nq^3 boxes; steps address sub-slices of each box.
    	// Every step assigns its full output sub-slice, so no zeroing is needed.
    	T scratch[2 * nq * nq * nq];
    	const ndview<T, nq, nq, nq> wsp0{scratch};
    	const ndview<T, nq, nq, nq> wsp1{scratch + nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        const nq_cview e_JxW{JxW + e * nq_cview::size};

        // step-1 : copy in -> wsp0 (nm^3 sub-block of the nq^3 box)
        for (index_t i = 0; i < nm; ++i)
            for (index_t j = 0; j < nm; ++j)
                for (index_t k = 0; k < nm; ++k)
                    wsp0(i, j, k) = e_in(i, j, k);

        // step-2 : direction 0
        for (index_t p = 0; p < nq; ++p)
            for (index_t k = 0; k < nm; ++k)
                for (index_t j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; ++i)
                        tmp += wsp0(i, j, k) * B(i, p);
                    wsp1(p, j, k) = tmp;
                }

        // step-3 : direction 1
        for (index_t q = 0; q < nq; ++q)
            for (index_t p = 0; p < nq; ++p)
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; ++j)
                        tmp += wsp1(p, j, k) * B(j, q);
                    wsp0(q, p, k) = tmp;
                }

        // step-4 : direction 2
        for (index_t r = 0; r < nq; ++r)
            for (index_t q = 0; q < nq; ++q)
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; ++k)
                        tmp += wsp0(q, p, k) * B(k, r);
                    wsp1(p, q, r) = tmp;
                }

        // Reverse operations

        // step-5 : multiply with weights and determinant of Jacobi
        for (index_t r = 0; r < nq; ++r)
            for (index_t q = 0; q < nq; ++q)
                for (index_t p = 0; p < nq; ++p)
                    wsp1(p, q, r) *= e_JxW(p, q, r);

        // step-6 : direction 2
        for (index_t k = 0; k < nm; ++k)
            for (index_t q = 0; q < nq; ++q)
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r)
                        tmp += wsp1(p, q, r) * B(k, r);
                    wsp0(q, p, k) = tmp;
                }

        // step-7 : direction 1
        for (index_t j = 0; j < nm; ++j)
            for (index_t k = 0; k < nm; ++k)
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q)
                        tmp += wsp0(q, p, k) * B(j, q);
                    wsp1(p, j, k) = tmp;
                }

        // step-8 : direction 0
        for (index_t i = 0; i < nm; ++i)
            for (index_t j = 0; j < nm; ++j)
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p)
                        tmp += wsp1(p, j, k) * B(i, p);
                    wsp0(i, j, k) = tmp;
                }

        // step-9 : copy wsp0 -> out
        for (index_t i = 0; i < nm; ++i)
            for (index_t j = 0; j < nm; ++j)
                for (index_t k = 0; k < nm; ++k)
                    e_out(i, j, k) = wsp0(i, j, k);
    }
}

} // namespace bk::bk1

#endif // BK1_KERNEL_H
