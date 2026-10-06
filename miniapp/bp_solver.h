#ifndef BP_SOLVER_H
#define BP_SOLVER_H

// Preconditioned conjugate gradients on the device.  The only preconditioner
// is the diagonal (Jacobi) one; `dinv == nullptr` runs plain CG, which is
// how the CEED bake-off problems are defined.

#include <chrono>
#include <cmath>
#include <cstddef>

#include "bp_backend.h"

namespace bp {

struct CGResult {
    int iterations = 0;
    double relative_residual = 0.0;   // ||r|| / ||b||
    double seconds = 0.0;
    bool converged = false;
};

template <typename T>
struct CGWorkspace {
    T* r = nullptr;
    T* z = nullptr;
    T* p = nullptr;
    T* Ap = nullptr;
};

// Solves A x = b with A = apply(cM, cK, .) starting from the given x.
template <typename T, typename Op>
CGResult pcg(Op& A, const T cM, const T cK, const T* dinv, const T* b, T* x,
             const CGWorkspace<T>& w, const double rtol, const int maxit)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const std::size_t n = A.nL;
    CGResult res;

    // r = b - A x
    A.apply(cM, cK, x, w.Ap);
    backend::lincomb(n, T(1), b, T(-1), w.Ap, w.r);
    const double bnorm = std::sqrt(double(backend::dot(n, b, b)));
    if (bnorm == 0.0) {
        res.converged = true;
        res.seconds = std::chrono::duration<double>(clock::now() - t0).count();
        return res;
    }
    const auto precondition = [&](const T* r, T* z) {
        if (dinv != nullptr) {
            backend::pointwise(n, dinv, r, z);
        } else {
            backend::copy(n, r, z);
        }
    };
    precondition(w.r, w.z);
    backend::copy(n, w.z, w.p);
    T rz = backend::dot(n, w.r, w.z);
    double rnorm = std::sqrt(double(backend::dot(n, w.r, w.r)));
    res.relative_residual = rnorm / bnorm;

    for (int it = 0; it < maxit; ++it) {
        if (res.relative_residual < rtol) {
            res.converged = true;
            break;
        }
        A.apply(cM, cK, w.p, w.Ap);
        const T alpha = rz / backend::dot(n, w.p, w.Ap);
        backend::axpby(n, alpha, w.p, T(1), x);
        backend::axpby(n, -alpha, w.Ap, T(1), w.r);
        ++res.iterations;
        rnorm = std::sqrt(double(backend::dot(n, w.r, w.r)));
        res.relative_residual = rnorm / bnorm;
        precondition(w.r, w.z);
        const T rz_new = backend::dot(n, w.r, w.z);
        const T beta = rz_new / rz;
        rz = rz_new;
        backend::axpby(n, T(1), w.z, beta, w.p);   // p = z + beta p
    }
    if (res.relative_residual < rtol) {
        res.converged = true;
    }
    res.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    return res;
}

} // namespace bp

#endif // BP_SOLVER_H
