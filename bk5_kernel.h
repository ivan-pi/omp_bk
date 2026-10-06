#ifndef BK5_KERNEL_H
#define BK5_KERNEL_H

// BK5 element kernel, shared by the BK5 benchmark driver (BK5.cpp) and the
// bp mini-app (miniapp/). Each kernel runs one OpenMP target region over the
// elements of an E-vector; the map clauses are no-ops when the arrays are
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
    #pragma omp target \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq*nq*nq], e_to_l[:nelmt*nq*nq*nq], L_in[:nL]) \
        map(tofrom: L_out[:nL])
    #pragma omp teams distribute parallel for
    for (std::size_t e = 0; e < nelmt; ++e) {

        // D(i, n) == dbasis[i * nq + n].
        // Constructed inside the target region on purpose: out here the view
        // would capture the *host* pointer (firstprivate structs get no
        // pointer translation), and OpenMP forbids statements between
        // `target` and `teams`, so per-iteration construction is the one
        // correct spot. It costs nothing -- it is a pointer copy the
        // compiler hoists.
        const ndview<const T, nq, nq> D{dbasis};

        // L-vector entry of each local node: the element's slice of e_to_l
        using idx_cview = ndview<const int, nq, nq, nq>;
        const idx_cview e_idx{e_to_l + e * idx_cview::size};

        // Gather the element's input box; the derivative loops below read
        // it nq times per entry, so it lives in local memory.
        T s_in[nq * nq * nq];
        const ndview<T, nq, nq, nq> e_in{s_in};
        for (index_t i = 0; i < nq; ++i) {
            for (index_t j = 0; j < nq; ++j) {
                for (index_t k = 0; k < nq; ++k) {
                    e_in(i, j, k) = L_in[e_idx(i, j, k)];
                }
            }
        }
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

                    #pragma omp atomic update
                    L_out[e_idx(i, j, k)] += alpha * tmp0;
                }
            }
        }
    }
}

} // namespace bk5
} // namespace bk

#endif // BK5_KERNEL_H
