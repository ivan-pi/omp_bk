#ifndef BK3_SUMFACT_H
#define BK3_SUMFACT_H

// bk3_sumfact.h -- the BK3 (Poisson operator) sum factorization: one element
// per iteration of a teams loop, private work arrays.  BK3.cpp's kernel, and
// the ref variant of openmp/offload/bk3_offload.cpp.
//   in(e,i,j,k) -> interpolate -> derivatives with D(n,p) -> fluxes with the
//   symmetric G(e,f,p,q,r) (factor-major) -> transposed derivatives with
//   D(p,n) -> interpolate back -> out(e,i,j,k)

#include <cstddef>

#include "bk_common.h"

namespace bk {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    constexpr int nm = nq - 1;
    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq*nq*nq], in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {
        // views built inside the target region: one built outside would hold
        // the host pointer (structs get no pointer translation)
        const ndview<const T, nm, nq> B{basis};    // B(i, p)
        const ndview<const T, nq, nq> D{dbasis};   // D(n, p)

        // five nq^3 boxes; every step writes its whole output sub-box
        T scratch[5 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + 1 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqr {scratch + 2 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqs {scratch + 3 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqt {scratch + 4 * nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        const ndview<const T, 6, nq, nq, nq> e_G{G + e * (6 * nq * nq * nq)};

        // 1: in -> wsp0
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    wsp0(i, j, k) = e_in(i, j, k);
                }
            }
        }
        // 2-4: interpolate along i, j, k -> u = wsp1(p, q, r)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; ++i) {
                        tmp += wsp0(i, j, k) * B(i, p);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; ++j) {
                        tmp += wsp1(p, j, k) * B(j, q);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; ++k) {
                        tmp += wsp0(q, p, k) * B(k, r);
                    }
                    wsp1(p, q, r) = tmp;
                }
            }
        }
        // 5-7: derivatives qr, qs, qt along p, q, r and the fluxes, each
        // diagonal factor with its own direction (Grr qr, Gss qs, Gtt qt)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {
                    const T Grr = e_G(0, p, q, r);
                    const T Grs = e_G(1, p, q, r);
                    const T Grt = e_G(2, p, q, r);
                    const T Gss = e_G(3, p, q, r);
                    const T Gst = e_G(4, p, q, r);
                    const T Gtt = e_G(5, p, q, r);
                    T qr = 0;
                    T qs = 0;
                    T qt = 0;
                    for (index_t n = 0; n < nq; ++n) {
                        qr += wsp1(n, q, r) * D(n, p);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qs += wsp1(p, n, r) * D(n, q);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qt += wsp1(p, q, n) * D(n, r);
                    }
                    rqr(p, q, r) = Grr * qr + Grs * qs + Grt * qt;
                    rqs(p, q, r) = Grs * qr + Gss * qs + Gst * qt;
                    rqt(p, q, r) = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }
        // 8: transposed derivatives -> wsp1
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {
                    T tmp0 = 0;
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqr(n, q, r) * D(p, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqs(p, n, r) * D(q, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqt(p, q, n) * D(r, n);
                    }
                    wsp1(p, q, r) = tmp0;
                }
            }
        }
        // 9-11: interpolate back along r, q, p -> wsp0(i, j, k)
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r) {
                        tmp += wsp1(p, q, r) * B(k, r);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q) {
                        tmp += wsp0(q, p, k) * B(j, q);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p) {
                        tmp += wsp1(p, j, k) * B(i, p);
                    }
                    wsp0(i, j, k) = tmp;
                }
            }
        }
        // 12: wsp0 -> out
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out(i, j, k) = wsp0(i, j, k);
                }
            }
        }
    }
}

} // namespace bk

#endif // BK3_SUMFACT_H
