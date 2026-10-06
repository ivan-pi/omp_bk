#ifndef BP_SOLVER_H
#define BP_SOLVER_H

// Preconditioned conjugate gradients on the device.  The only preconditioner
// is the diagonal (Jacobi) one; `dinv == nullptr` runs plain CG, which is
// how the CEED bake-off problems are defined.

#include <cmath>
#include <cstddef>

#include "bp_backend.h"
#include "bp_timer.h"

namespace bp {

struct CGResult {
    int iterations = 0;
    double relative_residual = 0.0;   // ||r|| / ||b||
    double seconds = 0.0;
    bool converged = false;           // also true after exactly maxit iterations with rtol <= 0
};

template <typename T>
struct CGWorkspace {
    T* r = nullptr;
    T* z = nullptr;
    T* p = nullptr;
    T* Ap = nullptr;
};

// Solves A x = b with A = apply(cM, cK, .) starting from x, or from x = 0
// without applying the operator to it when zero_guess is set (x must then
// hold zeros).
template <typename T, typename Op>
CGResult pcg(Op& A, const T cM, const T cK, const T* dinv, const T* b, T* x,
             const CGWorkspace<T>& w, const double rtol, const int maxit,
             const bool zero_guess)
{
    Stopwatch clock;
    clock.start();
    const std::size_t n = A.nL;
    CGResult res;

    // r = b - A x
    if (zero_guess) {
        backend::copy(n, b, w.r);
    } else {
        A.apply(cM, cK, x, w.Ap);
        backend::lincomb(n, T(1), b, T(-1), w.Ap, w.r);
    }
    const double bnorm = backend::norm(n, b);
    if (bnorm == 0.0) {
        res.converged = true;
        clock.stop();
        res.seconds = clock.seconds;
        return res;
    }

    // z = D^{-1} r; without a preconditioner z is r itself
    T* const z = (dinv != nullptr) ? w.z : w.r;
    const auto precondition = [&]() {
        if (dinv != nullptr) {
            backend::pointwise(n, T(1), dinv, w.r, w.z);
        }
    };
    precondition();
    backend::copy(n, z, w.p);
    T rz = backend::dot(n, w.r, z);
    T rr = (dinv != nullptr) ? backend::dot(n, w.r, w.r) : rz;
    res.relative_residual = std::sqrt(double(rr)) / bnorm;

    while (res.iterations < maxit && res.relative_residual >= rtol) {
        A.apply(cM, cK, w.p, w.Ap);
        const T alpha = rz / backend::dot(n, w.p, w.Ap);
        rr = backend::cg_update(n, alpha, w.p, w.Ap, x, w.r);
        ++res.iterations;
        res.relative_residual = std::sqrt(double(rr)) / bnorm;
        precondition();
        const T rz_new = (dinv != nullptr) ? backend::dot(n, w.r, z) : rr;
        const T beta = rz_new / rz;
        rz = rz_new;
        backend::axpby(n, T(1), z, beta, w.p);   // p = z + beta p
    }
    res.converged = rtol <= 0.0 || res.relative_residual < rtol;
    clock.stop();
    res.seconds = clock.seconds;
    return res;
}

} // namespace bp

#endif // BP_SOLVER_H
