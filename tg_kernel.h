#ifndef TG_KERNEL_H
#define TG_KERNEL_H

// Taylor-Galerkin (Lax-Wendroff) advection kernel for a constant velocity e,
// in the style of BK3: the right-hand side of one explicit step of
// u_t + e.grad u = 0,
//
//   r = int phi  a (e.grad u)  +  int (e.grad phi)  c (e.grad u),
//   a = -dt,  c = -dt^2 / 2,            M (u^{n+1} - u^n) = r,
//
// (the second-order term of the Taylor expansion integrated by parts; the
// wall surface term is not included). The velocity enters only through its
// reference-space image et = J^{-1} e, constant on affine (here Cartesian)
// elements, so unlike BK3 no metric data is read per quadrature point: the
// only per-element data are the nm^3 inputs and outputs. e.grad u is a
// combination of the three reference derivatives weighted by et; a
// direction with a zero component skips that contraction entirely, which is
// what makes lattice directions cheap.
//
// nm = nq - 1 is the Gauss-Legendre family (BK1/BK3: interpolate from the
// GLL nodes to the quadrature points and back); nm = nq the collocated
// family (BK5: the nodes are the quadrature points, B = I, so the
// interpolation and projection steps are skipped).
//
// Shared by the bp mini-app (miniapp/); the map clauses are no-ops when the
// arrays are already present in an enclosing target data region.

#include <cstddef>

#include "bk_common.h"

namespace bk {
namespace tg {

template <typename T, int nq, int nm = nq - 1, typename index_t = int>
void TaylorGalerkin(
    const std::size_t nelmt,
    const T* __restrict__ basis,     // B(i, p) = l_i(xq_p), nm x nq, as BK1/BK3; unused if nm == nq
    const T* __restrict__ dbasis,    // D(n, p) = L_n'(xq_p), nq x nq, node-major as BK3
    const T* __restrict__ weights,   // 1-D quadrature weights, nq
    const T detJ,                    // |J| of the affine element
    const T et0, const T et1, const T et2,   // et = J^{-1} e
    const T a,                       // value coefficient
    const T c,                       // flux coefficient
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(nm == nq - 1 || nm == nq, "Gauss-Legendre or collocated family");
    constexpr bool collocated = (nm == nq);

    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;

    #pragma omp target \
        map(to: basis[:nm*nq], dbasis[:nq*nq], weights[:nq]) \
        map(to: in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // Views built inside the target region (see BK3 for why)
        const ndview<const T, nm, nq> B{basis};
        const ndview<const T, nq, nq> D{dbasis};

        // Work arrays: three nq^3 boxes; every step assigns its full output
        // sub-slice, so no zeroing is needed.
        T scratch[3 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + 1 * nq * nq * nq};
        const ndview<T, nq, nq, nq> flux{scratch + 2 * nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};

        if constexpr (collocated) {
            // the nodes are the quadrature points: u(p, q, r) is the input
            for (index_t i = 0; i < nq; ++i) {
                for (index_t j = 0; j < nq; ++j) {
                    for (index_t k = 0; k < nq; ++k) {
                        wsp1(i, j, k) = e_in(i, j, k);
                    }
                }
            }
        } else {
        // step-1 : copy in -> wsp0 (nm^3 sub-block of the nq^3 box)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    wsp0(i, j, k) = e_in(i, j, k);
                }
            }
        }

        // steps 2-4 : interpolate to the quadrature points, one direction at
        // a time (as BK1); the result u(p, q, r) lands in wsp1
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
        }

        // step-5 : the point operation. s = e.grad u = et . grad_xi u at every
        // quadrature point, times the quadrature weight |J| w_p w_q w_r:
        // flux(p, q, r) = JxW s. Both weak-form terms are multiples of it.
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {
                    T s = 0;
                    if (et0 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += wsp1(n, q, r) * D(n, p);
                        }
                        s += et0 * d;
                    }
                    if (et1 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += wsp1(p, n, r) * D(n, q);
                        }
                        s += et1 * d;
                    }
                    if (et2 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += wsp1(p, q, n) * D(n, r);
                        }
                        s += et2 * d;
                    }
                    flux(p, q, r) = detJ * weights[p] * weights[q] * weights[r] * s;
                }
            }
        }

        // step-6 : integrate values and gradients in the quadrature-point
        // space: a (value term) plus c times the transposed directional
        // derivative (flux term, as BK3's step 8), result in wsp1
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {
                    T g = 0;
                    if (et0 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += flux(n, q, r) * D(p, n);
                        }
                        g += et0 * d;
                    }
                    if (et1 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += flux(p, n, r) * D(q, n);
                        }
                        g += et1 * d;
                    }
                    if (et2 != T(0)) {
                        T d = 0;
                        for (index_t n = 0; n < nq; ++n) {
                            d += flux(p, q, n) * D(r, n);
                        }
                        g += et2 * d;
                    }
                    wsp1(p, q, r) = a * flux(p, q, r) + c * g;
                }
            }
        }

        if constexpr (collocated) {
            for (index_t i = 0; i < nq; ++i) {
                for (index_t j = 0; j < nq; ++j) {
                    for (index_t k = 0; k < nq; ++k) {
                        e_out(i, j, k) = wsp1(i, j, k);
                    }
                }
            }
        } else {
        // steps 7-9 : project back to the nodes with B^T, one direction at a
        // time (as BK1's steps 6-8)
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

        // step-10 : copy wsp0 -> out
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out(i, j, k) = wsp0(i, j, k);
                }
            }
        }
        }
    }
}

} // namespace tg
} // namespace bk

#endif // TG_KERNEL_H
