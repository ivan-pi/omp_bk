#ifndef BK5_KERNEL_H
#define BK5_KERNEL_H

#include <cstddef>

#include "bk_common.h"

// ---------------------------------------------------------------------------
// BK5 kernel: scalar Laplace operator at quadrature points.
// D(i, n) is the 1-D derivative matrix, shared by all three directions;
// G holds six symmetric metric factors per point, interleaved as
// G(i, j, factor, k) within each element.
// Header-only so the timing driver (BK5.cpp) and the arithmetic-intensity
// tool can share one definition of the kernel.
// ---------------------------------------------------------------------------
namespace bk::bk5 {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    #pragma omp target \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq*nq*nq], in[:nelmt*nq*nq*nq]) \
        map(from: out[:nelmt*nq*nq*nq])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // D(i, n) == dbasis[i * nq + n].
        // Constructed inside the target region on purpose: out here the view
        // would capture the *host* pointer (firstprivate structs get no
        // pointer translation), and OpenMP forbids statements between
        // `target` and `teams`, so per-iteration construction is the one
        // correct spot. It costs nothing -- it is a pointer copy the
        // compiler hoists.
        const ndview<const T, nq, nq> D{dbasis};

        using nq_cview = ndview<const T, nq, nq, nq>;
        const nq_cview e_in{in + e * nq_cview::size};
        const ndview<T, nq, nq, nq> e_out{out + e * nq_cview::size};
        // factor index interleaved between j and k
        const ndview<const T, nq, nq, 6, nq> e_G{G + e * (6 * nq * nq * nq)};

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
}

} // namespace bk::bk5

#endif // BK5_KERNEL_H
