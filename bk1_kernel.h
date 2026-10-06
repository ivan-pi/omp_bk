#ifndef BK1_KERNEL_H
#define BK1_KERNEL_H

// BK1 element kernel, shared by the BK1 benchmark driver (BK1.cpp) and the
// bp mini-app (miniapp/). ElementKernel is the per-element computation;
// SumFactorization runs it in one OpenMP target region over the elements
// of an E-vector, SumFactorizationFused gathers from and scatters to an
// L-vector on the fly. The map clauses are no-ops when the arrays are
// already present in an enclosing target data region.

#include <cstddef>

#include "bk_common.h"

namespace bk {
namespace bk1 {

// The element kernel: one element's input box (E-vector slice or a
// gathered copy) to its output box. Called from the target regions below,
// hence `declare target`; the views are built here from the pointer
// arguments, which are device addresses inside a target region (a view
// built on the host would carry a host pointer onto the device).
#pragma omp declare target
template <typename T, int nq, typename index_t = int>
void ElementKernel(
    const T* __restrict__ basis,
    const T* __restrict__ JxW_e,
    const T* __restrict__ in_e,
    T* __restrict__ out_e)
{
    constexpr int nm = nq - 1;

    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;
    using nq_cview = ndview<const T, nq, nq, nq>;

    // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
    const ndview<const T, nm, nq> B{basis};

	// Work arrays: two nq^3 boxes; steps address sub-slices of each box.
	// Every step assigns its full output sub-slice, so no zeroing is needed.
	T scratch[2 * nq * nq * nq];
	const ndview<T, nq, nq, nq> wsp0{scratch};
	const ndview<T, nq, nq, nq> wsp1{scratch + nq * nq * nq};

    const nm_cview e_in {in_e};
    const nm_view  e_out{out_e};
    const nq_cview e_JxW{JxW_e};

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
#pragma omp end declare target

// One target region over the elements of an E-vector (the BK1 benchmark).
template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nm3 = std::size_t(nm) * nm * nm;
    constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {
        ElementKernel<T, nq, index_t>(basis, JxW + e * nq3, in + e * nm3, out + e * nm3);
    }
}

// The same operator without a stored E-vector: every element gathers its
// input from the L-vector through e_to_l and adds alpha times its output
// back with atomic updates,  L_out += alpha * P^T M_e P L_in.
// (`atomic` may not nest inside a `loop` region, hence the explicit form.)
template <typename T, int nq, typename index_t = int>
void SumFactorizationFused(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const int* __restrict__ e_to_l,
    const T alpha,
    const std::size_t nL,
    const T* __restrict__ L_in,
    T* __restrict__ L_out)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nm3 = std::size_t(nm) * nm * nm;
    constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    #pragma omp target teams distribute parallel for \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], e_to_l[:nelmt*nm3], L_in[:nL]) \
        map(tofrom: L_out[:nL])
    for (std::size_t e = 0; e < nelmt; ++e) {
        const int* idx = e_to_l + e * nm3;
        T s_in[nm3];
        T s_out[nm3];
        for (std::size_t s = 0; s < nm3; ++s) {
            s_in[s] = L_in[idx[s]];
        }
        ElementKernel<T, nq, index_t>(basis, JxW + e * nq3, s_in, s_out);
        for (std::size_t s = 0; s < nm3; ++s) {
            #pragma omp atomic update
            L_out[idx[s]] += alpha * s_out[s];
        }
    }
}

} // namespace bk1
} // namespace bk

#endif // BK1_KERNEL_H
