#ifndef BP_BASIS_H
#define BP_BASIS_H

// One-dimensional ingredients of the tensor-product spectral element:
// Gauss-Lobatto-Legendre (GLL) nodes carrying the degrees of freedom,
// Gauss-Legendre (GL) quadrature points, and the Lagrange interpolation and
// differentiation matrices between them, in the layouts the BK kernels read.
//
// Everything here runs once at setup on the host, in double precision, and
// is converted to the working precision at the end.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bp {

// Legendre polynomial P_n(x) and its derivative by the three-term recurrence.
inline void legendre(const int n, const double x, double& p, double& dp)
{
    double p0 = 1.0;
    double p1 = x;
    if (n == 0) {
        p = 1.0;
        dp = 0.0;
        return;
    }
    for (int k = 2; k <= n; ++k) {
        const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
    }
    p = p1;
    // (x^2 - 1) P_n' = n (x P_n - P_{n-1})
    dp = n * (x * p1 - p0) / (x * x - 1.0);
}

struct Rule {
    std::vector<double> x;   // points in [-1, 1], ascending
    std::vector<double> w;   // quadrature weights
};

// Gauss-Legendre rule with n points: the roots of P_n.
inline Rule gauss_legendre(const int n)
{
    assert(n >= 1);
    Rule r;
    r.x.resize(n);
    r.w.resize(n);
    for (int i = 0; i < n; ++i) {
        // Chebyshev-like initial guess, refined by Newton on P_n
        double x = -std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double p = 0.0;
        double dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            legendre(n, x, p, dp);
            const double dx = p / dp;
            x -= dx;
            if (std::fabs(dx) < 1e-15) {
                break;
            }
        }
        legendre(n, x, p, dp);
        r.x[i] = x;
        r.w[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    return r;
}

// Gauss-Lobatto-Legendre rule with n points: +-1 and the roots of P_{n-1}'.
inline Rule gauss_lobatto_legendre(const int n)
{
    assert(n >= 2);
    Rule r;
    r.x.resize(n);
    r.w.resize(n);
    const int m = n - 1;
    r.x[0] = -1.0;
    r.x[m] = 1.0;
    for (int i = 1; i < m; ++i) {
        double x = -std::cos(M_PI * i / m);
        for (int it = 0; it < 100; ++it) {
            // Newton on P_m'(x) = 0 using  (1-x^2) P_m'' = 2x P_m' - m(m+1) P_m
            double p = 0.0;
            double dp = 0.0;
            legendre(m, x, p, dp);
            const double ddp = (2.0 * x * dp - m * (m + 1.0) * p) / (1.0 - x * x);
            const double dx = dp / ddp;
            x -= dx;
            if (std::fabs(dx) < 1e-15) {
                break;
            }
        }
        r.x[i] = x;
    }
    for (int i = 0; i < n; ++i) {
        double p = 0.0;
        double dp = 0.0;
        legendre(m, r.x[i], p, dp);
        r.w[i] = 2.0 / (m * (m + 1.0) * p * p);
    }
    return r;
}

// Lagrange basis through the nodes `nodes`, evaluated at `points`:
// interp[i * npts + p] = l_i(x_p) and deriv[i * npts + p] = l_i'(x_p).
// This "node-major" layout is what BK1/BK3 read as B(i, p).
inline void lagrange_matrices(const std::vector<double>& nodes,
                              const std::vector<double>& points,
                              std::vector<double>& interp,
                              std::vector<double>& deriv)
{
    const int nn = static_cast<int>(nodes.size());
    const int np = static_cast<int>(points.size());
    interp.assign(std::size_t(nn) * np, 0.0);
    deriv.assign(std::size_t(nn) * np, 0.0);
    for (int i = 0; i < nn; ++i) {
        for (int p = 0; p < np; ++p) {
            const double x = points[p];
            double li = 1.0;
            double dli = 0.0;
            for (int j = 0; j < nn; ++j) {
                if (j == i) {
                    continue;
                }
                // product rule: d/dx prod_j (x - x_j)/(x_i - x_j)
                double term = 1.0 / (nodes[i] - nodes[j]);
                for (int k = 0; k < nn; ++k) {
                    if (k != i && k != j) {
                        term *= (x - nodes[k]) / (nodes[i] - nodes[k]);
                    }
                }
                dli += term;
                li *= (x - nodes[j]) / (nodes[i] - nodes[j]);
            }
            interp[std::size_t(i) * np + p] = li;
            deriv[std::size_t(i) * np + p] = dli;
        }
    }
}

// The 1-D operators of one discretisation, shared by the three directions.
//
//   nm   nodes (degrees of freedom) per direction, nm = p + 1
//   nq   quadrature points per direction: p + 2 (Gauss-Legendre, BK1/BK3)
//        or p + 1 (collocated Gauss-Lobatto-Legendre, BK5)
//
//   B  [nm x nq]  B(i, p)  = l_i(xq_p)      -- BK1/BK3 `basis`
//   DB [nm x nq]  DB(i, p) = l_i'(xq_p)     -- derivative of the nodal basis
//                                             at the quadrature points
//   Dq [nq x nq]  Dq(n, p) = L_n'(xq_p)     -- derivative of the quadrature
//                                             point Lagrange basis; BK3 reads
//                                             it as D(n, p), BK5 as D(p, n)
struct Basis1D {
    int nm = 0;
    int nq = 0;
    bool collocated = false;
    Rule nodes;        // GLL nodes and weights
    Rule quad;         // quadrature points and weights
    std::vector<double> B;
    std::vector<double> DB;
    std::vector<double> Dq;       // node-major (n, p), BK3's layout
    std::vector<double> Dq_T;     // point-major (p, n), BK5's layout
};

inline Basis1D make_basis(const int p, const bool collocated)
{
    assert(p >= 1);
    Basis1D b;
    b.nm = p + 1;
    b.collocated = collocated;
    b.nodes = gauss_lobatto_legendre(b.nm);
    if (collocated) {
        b.nq = p + 1;
        b.quad = b.nodes;
    } else {
        b.nq = p + 2;
        b.quad = gauss_legendre(b.nq);
    }
    lagrange_matrices(b.nodes.x, b.quad.x, b.B, b.DB);
    std::vector<double> unused;
    lagrange_matrices(b.quad.x, b.quad.x, unused, b.Dq);
    b.Dq_T.assign(b.Dq.size(), 0.0);
    for (int n = 0; n < b.nq; ++n) {
        for (int q = 0; q < b.nq; ++q) {
            b.Dq_T[std::size_t(q) * b.nq + n] = b.Dq[std::size_t(n) * b.nq + q];
        }
    }
    return b;
}

template <typename T>
std::vector<T> to_precision(const std::vector<double>& v)
{
    return std::vector<T>(v.begin(), v.end());
}

} // namespace bp

#endif // BP_BASIS_H
