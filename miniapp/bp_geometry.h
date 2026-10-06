#ifndef BP_GEOMETRY_H
#define BP_GEOMETRY_H

// Metric factors at the quadrature points of every element, stored in the
// layouts the kernels read, plus the diagonals of the element mass and
// stiffness matrices for the Jacobi preconditioner.
//
// With J = dx/dxi the Jacobian of the reference-to-physical map and w the
// tensor-product quadrature weight,
//
//     JxW  = |J| w                            mass     (BK1, lumped GLL mass)
//     G    = |J| w  J^{-1} J^{-T}  (symmetric) stiffness (BK3, BK5)
//
// so that  u^T K v = sum_q (grad_xi u)^T G (grad_xi v).  More generally
// G = |J| w J^{-1} C J^{-T} for a constant symmetric coefficient tensor C:
// the identity gives the Laplacian, C = e e^T the directional stiffness
// int (e.grad u)(e.grad v) of the Taylor-Galerkin transport step, and the
// same BK3/BK5 kernels apply either.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "bp_basis.h"
#include "bp_mesh.h"

namespace bp {

template <typename T>
struct Geometry {
    std::vector<T> JxW;       // (e, p, q, r)                 nelmt * nq^3
    std::vector<T> G;         // BK3: (e, factor, p, q, r)    nelmt * 6 nq^3
                              // BK5: (e, p, q, factor, r)
    std::vector<T> diag_mass; // E-vector layout (e, i, j, k) nelmt * nm^3
    std::vector<T> diag_stiff;
};

using Tensor3 = std::array<std::array<double, 3>, 3>;

inline Tensor3 identity_tensor()
{
    return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

inline Tensor3 outer_product(const std::array<double, 3>& e)
{
    Tensor3 C{};
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) {
            C[a][c] = e[a] * e[c];
        }
    }
    return C;
}

// out(i, j, k) += scale * sum_{p,q,r} A(i, p) B(j, q) C(k, r) w(p, q, r), with
// A, B, C node-major [nm x nq] and w an nq^3 box: the sum-factorised
// evaluation that the kernels perform, here on the host for the diagonals.
// t1 and t2 are scratch of nm nq^2 and nm^2 nq entries.
inline void contract3(const int nm, const int nq,
                      const double* A, const double* B, const double* C,
                      const double* w, const double scale, double* out,
                      std::vector<double>& t1, std::vector<double>& t2)
{
    t1.resize(std::size_t(nm) * nq * nq);
    t2.resize(std::size_t(nm) * nm * nq);
    for (int i = 0; i < nm; ++i) {
        for (int q = 0; q < nq; ++q) {
            for (int r = 0; r < nq; ++r) {
                double s = 0.0;
                for (int p = 0; p < nq; ++p) {
                    s += A[i * nq + p] * w[(p * nq + q) * nq + r];
                }
                t1[(i * nq + q) * nq + r] = s;
            }
        }
    }
    for (int i = 0; i < nm; ++i) {
        for (int j = 0; j < nm; ++j) {
            for (int r = 0; r < nq; ++r) {
                double s = 0.0;
                for (int q = 0; q < nq; ++q) {
                    s += B[j * nq + q] * t1[(i * nq + q) * nq + r];
                }
                t2[(i * nm + j) * nq + r] = s;
            }
        }
    }
    for (int i = 0; i < nm; ++i) {
        for (int j = 0; j < nm; ++j) {
            for (int k = 0; k < nm; ++k) {
                double s = 0.0;
                for (int r = 0; r < nq; ++r) {
                    s += C[k * nq + r] * t2[(i * nm + j) * nq + r];
                }
                out[(i * nm + j) * nm + k] += scale * s;
            }
        }
    }
}

template <typename T>
Geometry<T> make_geometry(const Mesh& m, const Basis1D& b,
                          const Tensor3& C = identity_tensor())
{
    const int nq = b.nq;
    const int nm = b.nm;
    const std::size_t nelmt = m.num_elements();
    const std::size_t nq3 = std::size_t(nq) * nq * nq;
    const std::size_t nm3 = std::size_t(nm) * nm * nm;

    Geometry<T> geo;
    geo.JxW.resize(nelmt * nq3);
    geo.G.resize(nelmt * 6 * nq3);
    geo.diag_mass.assign(nelmt * nm3, T(0));
    geo.diag_stiff.assign(nelmt * nm3, T(0));

    // 1-D matrices entering the diagonals: the diagonal of a tensor-product
    // operator B^T W B is (B o B)^T w, and the mixed derivative terms pair
    // B with DB.
    std::vector<double> BB(b.B.size());
    std::vector<double> DD(b.B.size());
    std::vector<double> BD(b.B.size());
    for (std::size_t s = 0; s < b.B.size(); ++s) {
        BB[s] = b.B[s] * b.B[s];
        DD[s] = b.DB[s] * b.DB[s];
        BD[s] = b.B[s] * b.DB[s];
    }

    #pragma omp parallel
    {
        // per-thread scratch, reused by every element
        std::vector<double> jxw(nq3);
        std::vector<double> g(6 * nq3);   // factor-major (f, p, q, r)
        std::vector<double> dm(nm3);
        std::vector<double> dk(nm3);
        std::vector<double> t1;
        std::vector<double> t2;

        #pragma omp for collapse(3)
        for (int ex = 0; ex < m.nelem[0]; ++ex) {
            for (int ey = 0; ey < m.nelem[1]; ++ey) {
                for (int ez = 0; ez < m.nelem[2]; ++ez) {
                    const std::array<int, 3> ec = {ex, ey, ez};
                    const std::size_t e = m.element_id(ex, ey, ez);
                    for (int p = 0; p < nq; ++p) {
                        for (int q = 0; q < nq; ++q) {
                            for (int r = 0; r < nq; ++r) {
                                const std::array<double, 3> xi = {b.quad.x[p], b.quad.x[q], b.quad.x[r]};
                                const std::array<double, 3> X = m.undeformed(ec, xi);
                                const auto Jp = m.physical_jacobian(X);
                                // J = d(physical)/dX * dX/dxi
                                double J[3][3];
                                for (int a = 0; a < 3; ++a) {
                                    for (int c = 0; c < 3; ++c) {
                                        J[a][c] = Jp[a][c] * m.half_h(c);
                                    }
                                }
                                const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
                                                 - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
                                                 + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
                                double Ji[3][3];   // inverse by cofactors
                                Ji[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / det;
                                Ji[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / det;
                                Ji[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / det;
                                Ji[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / det;
                                Ji[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / det;
                                Ji[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / det;
                                Ji[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / det;
                                Ji[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / det;
                                Ji[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / det;
                                const double wq = b.quad.w[p] * b.quad.w[q] * b.quad.w[r] * det;
                                const std::size_t idx = (std::size_t(p) * nq + q) * nq + r;
                                jxw[idx] = wq;
                                // JiC = J^{-1} C, then the symmetric factors
                                // (J^{-1} C J^{-T})_ac in the order rr, rs, rt, ss, st, tt
                                double JiC[3][3];
                                for (int a = 0; a < 3; ++a) {
                                    for (int c = 0; c < 3; ++c) {
                                        JiC[a][c] = 0.0;
                                        for (int d = 0; d < 3; ++d) {
                                            JiC[a][c] += Ji[a][d] * C[d][c];
                                        }
                                    }
                                }
                                const int pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
                                for (int f = 0; f < 6; ++f) {
                                    const int a = pairs[f][0];
                                    const int c = pairs[f][1];
                                    double s = 0.0;
                                    for (int d = 0; d < 3; ++d) {
                                        s += JiC[a][d] * Ji[c][d];
                                    }
                                    g[f * nq3 + idx] = wq * s;
                                }
                            }
                        }
                    }

                    // element diagonals
                    std::fill(dm.begin(), dm.end(), 0.0);
                    std::fill(dk.begin(), dk.end(), 0.0);
                    contract3(nm, nq, BB.data(), BB.data(), BB.data(), jxw.data(), 1.0, dm.data(), t1, t2);
                    contract3(nm, nq, DD.data(), BB.data(), BB.data(), g.data() + 0 * nq3, 1.0, dk.data(), t1, t2);
                    contract3(nm, nq, BD.data(), BD.data(), BB.data(), g.data() + 1 * nq3, 2.0, dk.data(), t1, t2);
                    contract3(nm, nq, BD.data(), BB.data(), BD.data(), g.data() + 2 * nq3, 2.0, dk.data(), t1, t2);
                    contract3(nm, nq, BB.data(), DD.data(), BB.data(), g.data() + 3 * nq3, 1.0, dk.data(), t1, t2);
                    contract3(nm, nq, BB.data(), BD.data(), BD.data(), g.data() + 4 * nq3, 2.0, dk.data(), t1, t2);
                    contract3(nm, nq, BB.data(), BB.data(), DD.data(), g.data() + 5 * nq3, 1.0, dk.data(), t1, t2);

                    // store in the kernel layouts
                    for (std::size_t s = 0; s < nq3; ++s) {
                        geo.JxW[e * nq3 + s] = T(jxw[s]);
                    }
                    T* const Ge = geo.G.data() + e * 6 * nq3;
                    if (b.collocated) {
                        // BK5 reads (p, q, factor, r)
                        for (int p = 0; p < nq; ++p) {
                            for (int q = 0; q < nq; ++q) {
                                for (int f = 0; f < 6; ++f) {
                                    for (int r = 0; r < nq; ++r) {
                                        const std::size_t src = f * nq3 + (std::size_t(p) * nq + q) * nq + r;
                                        Ge[((std::size_t(p) * nq + q) * 6 + f) * nq + r] = T(g[src]);
                                    }
                                }
                            }
                        }
                    } else {
                        // BK3 reads the factor-major layout as it is
                        for (std::size_t s = 0; s < 6 * nq3; ++s) {
                            Ge[s] = T(g[s]);
                        }
                    }
                    for (std::size_t s = 0; s < nm3; ++s) {
                        geo.diag_mass[e * nm3 + s] = T(dm[s]);
                        geo.diag_stiff[e * nm3 + s] = T(dk[s]);
                    }
                }
            }
        }
    }
    return geo;
}

} // namespace bp

#endif // BP_GEOMETRY_H
