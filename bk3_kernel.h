#ifndef BK3_KERNEL_H
#define BK3_KERNEL_H

#include <cstddef>

#include "bk_common.h"

namespace bk::bk3 {

// BK3: Poisson (stiffness) operator via sum factorization.
//   nq = number of quadrature points (per direction), nm = nq - 1 modes.
// Header-only so the timing driver (BK3.cpp) and the arithmetic-intensity
// tool can share one definition of the kernel.
template <typename T, int nq, int nm = nq - 1, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq*nq*nq], in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // Views onto the mapped basis matrices. Constructed *inside* the
        // target region: a view built outside would capture the host pointer
        // (firstprivate/implicitly-mapped structs get no pointer translation),
        // so B(i,p)/D(n,p) would dereference host memory on the device.
        // This is the portable form used by BK1 and BK5.
        //
        // An alternative is to build B/D on the host and add `map(to: B, D)`
        // to the target clause. Both g++ 13 and clang++ 18 compile that and it
        // runs correctly under host fallback, but whether the runtime attaches
        // the struct's pointer member to the mapped basis on a real device is
        // implementation-defined (reported valid for the Intel runtime); it
        // was not validated on a GPU here, so the construct-inside form is kept.
        // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
        const ndview<const T, nm, nq> B{basis};
        // D(row, col) == dbasis[row * nq + col], shared by all directions
        const ndview<const T, nq, nq> D{dbasis};

        // Work arrays: five nq^3 boxes; steps address sub-slices of each box.
        // Every step assigns its full output sub-slice, so no zeroing is needed.
        // (The reference implementation sized rqr/rqs/rqt at nelmt*nq^3 each,
        //  but only the current element's slice is ever live; one box each
        //  suffices serially.)
        T scratch[5 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + 1 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqr {scratch + 2 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqs {scratch + 3 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqt {scratch + 4 * nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        // G(factor, p, q, r): factor-major within each element
        const ndview<const T, 6, nq, nq, nq> e_G{G + e * (6 * nq * nq * nq)};

        /*
        Interpolate to GL nodes
        */

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

        // steps 5-7 : load geometric factors, multiply by D, apply chain rule.
        // The symmetric metric G is applied so that the diagonal factor pairs
        // with the same-direction derivative (Grr*qr, Gss*qs, Gtt*qt), matching
        // BK5. qr/qs/qt are the derivatives along directions 0/1/2.
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {

                    const T Grr = e_G(0, p, q, r);
                    const T Grs = e_G(1, p, q, r);
                    const T Grt = e_G(2, p, q, r);
                    const T Gss = e_G(3, p, q, r);
                    const T Gst = e_G(4, p, q, r);
                    const T Gtt = e_G(5, p, q, r);

                    T qr = 0, qs = 0, qt = 0;

                    for (index_t n = 0; n < nq; ++n)
                        qr += wsp1(n, q, r) * D(n, p);

                    for (index_t n = 0; n < nq; ++n)
                        qs += wsp1(p, n, r) * D(n, q);

                    for (index_t n = 0; n < nq; ++n)
                        qt += wsp1(p, q, n) * D(n, r);

                    rqr(p, q, r) = Grr * qr + Grs * qs + Grt * qt;
                    rqs(p, q, r) = Grs * qr + Gss * qs + Gst * qt;
                    rqt(p, q, r) = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        // step-8 : compute out vector in GL nodes
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {

                    T tmp0 = 0;
                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqr(n, q, r) * D(p, n);

                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqs(p, n, r) * D(q, n);

                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqt(p, q, n) * D(r, n);

                    wsp1(p, q, r) = tmp0;
                }
            }
        }

        /*
        Interpolate to GLL nodes
        */

        // step-9 : direction 2
        for (index_t k = 0; k < nm; ++k)
            for (index_t q = 0; q < nq; ++q)
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r)
                        tmp += wsp1(p, q, r) * B(k, r);
                    wsp0(q, p, k) = tmp;
                }

        // step-10 : direction 1
        for (index_t j = 0; j < nm; ++j)
            for (index_t k = 0; k < nm; ++k)
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q)
                        tmp += wsp0(q, p, k) * B(j, q);
                    wsp1(p, j, k) = tmp;
                }

        // step-11 : direction 0
        for (index_t i = 0; i < nm; ++i)
            for (index_t j = 0; j < nm; ++j)
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p)
                        tmp += wsp1(p, j, k) * B(i, p);
                    wsp0(i, j, k) = tmp;
                }

        // step-12 : copy wsp0 -> out
        for (index_t i = 0; i < nm; ++i)
            for (index_t j = 0; j < nm; ++j)
                for (index_t k = 0; k < nm; ++k)
                    e_out(i, j, k) = wsp0(i, j, k);
    }
}

} // namespace bk::bk3

#endif // BK3_KERNEL_H
