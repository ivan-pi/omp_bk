#ifndef BK5_KERNEL_H
#define BK5_KERNEL_H

// BK5 element kernel, shared by the BK5 benchmark driver (BK5.cpp) and the
// bp mini-app (miniapp/). ElementKernel is the per-element computation;
// SumFactorization runs it in one OpenMP target region over the elements
// of an E-vector, SumFactorizationFused gathers from and scatters to an
// L-vector on the fly. The map clauses are no-ops when the arrays are
// already present in an enclosing target data region.

#include <cstddef>

#include "bk_common.h"

// ---------------------------------------------------------------------------
// BK5 kernel: scalar Laplace operator at quadrature points.
// D(i, n) is the 1-D derivative matrix, shared by all three directions;
// G holds six symmetric metric factors per point, interleaved as
// G(i, j, factor, k) within each element.
// ---------------------------------------------------------------------------
namespace bk {
namespace bk5 {

// The element kernel: one element's input box (E-vector slice or a
// gathered copy) to its output box. Called from the target regions below,
// hence `declare target`; the views are built here from the pointer
// arguments, which are device addresses inside a target region (a view
// built on the host would carry a host pointer onto the device).
#pragma omp declare target
template <typename T, int nq, typename index_t = int>
void ElementKernel(
    const T* __restrict__ dbasis,
    const T* __restrict__ G_e,
    const T* __restrict__ in_e,
          T* __restrict__ out_e)
{
    // D(i, n) == dbasis[i * nq + n].
    const ndview<const T, nq, nq> D{dbasis};

    using nq_cview = ndview<const T, nq, nq, nq>;
    const nq_cview e_in{in_e};
    const ndview<T, nq, nq, nq> e_out{out_e};
    // factor index interleaved between j and k
    const ndview<const T, nq, nq, 6, nq> e_G{G_e};

    // Intermediate vals
    T s_rqr[nq * nq * nq];
    T s_rqs[nq * nq * nq];
    T s_rqt[nq * nq * nq];
    const ndview<T, nq, nq, nq> rqr{s_rqr};
    const ndview<T, nq, nq, nq> rqs{s_rqs};
    const ndview<T, nq, nq, nq> rqt{s_rqt};

    for (index_t i = 0; i < nq; ++i) {
        for (index_t j = 0; j < nq; ++j) {
            for (index_t k = 0; k < nq; ++k) {

                // Load geometric factors, coalesced access
                const T Grr = e_G(i, j, 0, k);
                const T Grs = e_G(i, j, 1, k);
                const T Grt = e_G(i, j, 2, k);
                const T Gss = e_G(i, j, 3, k);
                const T Gst = e_G(i, j, 4, k);
                const T Gtt = e_G(i, j, 5, k);

                // Multiply by D
                T qr = T(0), qs = T(0), qt = T(0);

                for (index_t n = 0; n < nq; ++n)
                    qr += e_in(n, j, k) * D(i, n);

                for (index_t n = 0; n < nq; ++n)
                    qs += e_in(i, n, k) * D(j, n);

                for (index_t n = 0; n < nq; ++n)
                    qt += e_in(i, j, n) * D(k, n);

                // Apply chain rule
                rqr(i, j, k) = Grr * qr + Grs * qs + Grt * qt;
                rqs(i, j, k) = Grs * qr + Gss * qs + Gst * qt;
                rqt(i, j, k) = Grt * qr + Gst * qs + Gtt * qt;
            }
        }
    }

    for (index_t i = 0; i < nq; ++i) {
        for (index_t j = 0; j < nq; ++j) {
            for (index_t k = 0; k < nq; ++k) {

                T tmp0 = T(0);
                for (index_t n = 0; n < nq; ++n)
                    tmp0 += rqr(n, j, k) * D(n, i);

                for (index_t n = 0; n < nq; ++n)
                    tmp0 += rqs(i, n, k) * D(n, j);

                for (index_t n = 0; n < nq; ++n)
                    tmp0 += rqt(i, j, n) * D(n, k);

                e_out(i, j, k) = tmp0;
            }
        }
    }
}
#pragma omp end declare target

// One target region over the elements of an E-vector (the BK5 benchmark).
template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    #pragma omp target \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq3], in[:nelmt*nq3]) \
        map(from: out[:nelmt*nq3])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {
        ElementKernel<T, nq, index_t>(dbasis, G + e * 6 * nq3, in + e * nq3, out + e * nq3);
    }
}

// The same operator without a stored E-vector: every element gathers its
// input from the L-vector through e_to_l and adds alpha times its output
// back with atomic updates,  L_out += alpha * P^T K_e P L_in.
// (`atomic` may not nest inside a `loop` region, hence the explicit form.)
template <typename T, int nq, typename index_t = int>
void SumFactorizationFused(
    const std::size_t nelmt,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const int* __restrict__ e_to_l,
    const T alpha,
    const std::size_t nL,
    const T* __restrict__ L_in,
    T* __restrict__ L_out)
{
    constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    #pragma omp target teams distribute parallel for \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq3], e_to_l[:nelmt*nq3], L_in[:nL]) \
        map(tofrom: L_out[:nL])
    for (std::size_t e = 0; e < nelmt; ++e) {
        const int* idx = e_to_l + e * nq3;
        T s_in[nq3];
        T s_out[nq3];
        for (std::size_t s = 0; s < nq3; ++s) {
            s_in[s] = L_in[idx[s]];
        }
        ElementKernel<T, nq, index_t>(dbasis, G + e * 6 * nq3, s_in, s_out);
        for (std::size_t s = 0; s < nq3; ++s) {
            #pragma omp atomic update
            L_out[idx[s]] += alpha * s_out[s];
        }
    }
}

} // namespace bk5
} // namespace bk

#endif // BK5_KERNEL_H
