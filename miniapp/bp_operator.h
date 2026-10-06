#ifndef BP_OPERATOR_H
#define BP_OPERATOR_H

// The assembled operator  A = cM M + cK K  applied matrix-free:
//
//     y = mask o P^T ( cM M_e + cK K_e ) P x
//
// with M_e / K_e the element mass / stiffness kernels (BK1 / BK3 for the
// Gauss-Legendre family; BK5 for the collocated family, whose lumped GLL
// mass is diagonal and is applied as an L-vector) and `mask` zeroing the
// Dirichlet boundary nodes.  The mass solve (cM = 1, cK = 0, no mask), the
// Poisson solve (cM = 0, cK = 1), the implicit diffusion step (cM = 1,
// cK = theta dt) and the Taylor-Galerkin solves are all instances of this
// one operator; `advect` applies the Taylor-Galerkin right-hand side kernel
// through the same restriction.  When the operator is diagonal (collocated
// family, cK = 0) the caller solves with the assembled diagonal directly.
//
// Two code paths apply it.  The E-vector path stores P x, runs each kernel
// on it and scatters the result with its coefficient (the structure of the
// CEED bake-off problems: gather, kernel, scatter as separate passes).  The
// fused path never stores an E-vector: each element gathers its input
// straight from x and adds its scaled output into y with atomics, so the
// kernels' in/out traffic to global memory disappears at the price of
// atomic updates and a run-to-run variation at round-off.

#include <cassert>
#include <cstddef>

#include "../bk1_kernel.h"
#include "../bk3_kernel.h"
#include "../bk5_kernel.h"
#include "../tg_kernel.h"
#include "bp_backend.h"
#include "bp_timer.h"

namespace bp {

template <typename T, int nq, bool collocated>
struct Operator {
    static constexpr int nm = collocated ? nq : nq - 1;
    static constexpr std::size_t nm3 = std::size_t(nm) * nm * nm;

    std::size_t nelmt = 0;
    std::size_t nL = 0;
    std::size_t nE = 0;

    // restriction (bp_mesh.h)
    const int* e_to_l = nullptr;
    const int* l_offsets = nullptr;
    const int* l_to_e = nullptr;
    bool atomic_scatter = false;
    bool fused = false;        // gather on the fly instead of an E-vector

    // basis and metric (bp_basis.h, bp_geometry.h) in the kernels' layouts
    const T* B = nullptr;        // BK1/BK3 basis (nm x nq); unused when collocated
    const T* D = nullptr;        // BK3: Dq (n, p); BK5: Dq^T (p, n)
    const T* Dq = nullptr;       // Dq (n, p) for the transport kernel in both families
    const T* JxW = nullptr;      // BK1 quadrature weights; unused when collocated
    const T* G = nullptr;
    const T* mass_diag = nullptr;   // collocated: the assembled lumped mass P^T JxW
    const T* mask = nullptr;        // nullptr: no Dirichlet condition

    // transport (tg_kernel.h on the Cartesian mesh)
    const T* w1d = nullptr;    // 1-D quadrature weights
    T detJ = 0;                // |J| of the (affine) elements
    T et[3] = {0, 0, 0};       // J^{-1} e

    // E-vector work space
    T* e_in = nullptr;
    T* e_out = nullptr;

    Stopwatch t_gather;
    Stopwatch t_mass;
    Stopwatch t_stiff;
    Stopwatch t_advect;
    Stopwatch t_scatter;
    Stopwatch t_mask;
    long applications = 0;

    double seconds() const
    {
        return t_gather.seconds + t_mass.seconds + t_stiff.seconds
             + t_advect.seconds + t_scatter.seconds + t_mask.seconds;
    }

    // ---- the timed phases ---------------------------------------------------

    void gather(const T* x)
    {
        t_gather.start();
        backend::gather(nL, nE, e_to_l, x, e_in);
        t_gather.stop();
    }

    // y = alpha P^T E (+ y)
    void scatter(const T alpha, const T* E, T* y, const bool accumulate)
    {
        t_scatter.start();
        if (atomic_scatter) {
            backend::scatter_add_atomic(nL, nE, e_to_l, alpha, E, y, accumulate);
        } else {
            backend::scatter_add(nL, nE, l_offsets, l_to_e, alpha, E, y, accumulate);
        }
        t_scatter.stop();
    }

    void apply_mask(T* y, const bool masked)
    {
        t_mask.start();
        if (masked && mask != nullptr) {
            backend::pointwise_inplace(nL, mask, y);
        }
        t_mask.stop();
    }

    // out = M_e in (Gauss-Legendre family only; the collocated mass is diagonal)
    void mass_e(const T* in, T* out)
    {
        assert(!collocated);
        t_mass.start();
        bk::bk1::SumFactorization<T, nq>(nelmt, B, JxW, in, out);
        t_mass.stop();
    }

    // out = K_e in
    void stiffness_e(const T* in, T* out)
    {
        t_stiff.start();
        if constexpr (collocated) {
            bk::bk5::SumFactorization<T, nq>(nelmt, D, G, in, out);
        } else {
            bk::bk3::SumFactorization<T, nq>(nelmt, B, D, G, in, out);
        }
        t_stiff.stop();
    }

    // y += alpha P^T K_e P x, gathering on the fly
    void stiffness_fused(const T alpha, const T* x, T* y)
    {
        t_stiff.start();
        if constexpr (collocated) {
            bk::bk5::SumFactorizationFused<T, nq>(nelmt, D, G, e_to_l, alpha, nL, x, y);
        } else {
            bk::bk3::SumFactorizationFused<T, nq>(nelmt, B, D, G, e_to_l, alpha, nL, x, y);
        }
        t_stiff.stop();
    }

    // y += alpha P^T M_e P x, gathering on the fly
    void mass_fused(const T alpha, const T* x, T* y)
    {
        t_mass.start();
        if constexpr (collocated) {
            backend::pointwise_add(nL, alpha, mass_diag, x, y);
        } else {
            bk::bk1::SumFactorizationFused<T, nq>(nelmt, B, JxW, e_to_l, alpha, nL, x, y);
        }
        t_mass.stop();
    }

    // ---- the operators ------------------------------------------------------

    // y = [mask o] P^T (cM M_e + cK K_e) P x
    void apply(const T cM, const T cK, const T* x, T* y, const bool masked = true)
    {
        assert(nE == nelmt * nm3);
        assert(cM != T(0) || cK != T(0));
        ++applications;
        if (fused) {
            t_scatter.start();   // the fused kernels accumulate into a zeroed y
            backend::fill(nL, T(0), y);
            t_scatter.stop();
            if (cM != T(0)) {
                mass_fused(cM, x, y);
            }
            if (cK != T(0)) {
                stiffness_fused(cK, x, y);
            }
        } else {
            bool have = false;   // y holds a partial result
            if (collocated && cM != T(0)) {
                t_mass.start();
                backend::pointwise(nL, cM, mass_diag, x, y);
                t_mass.stop();
                have = true;
            }
            if (cK != T(0) || !collocated) {
                gather(x);
            }
            if (!collocated && cM != T(0)) {
                mass_e(e_in, e_out);
                scatter(cM, e_out, y, have);
                have = true;
            }
            if (cK != T(0)) {
                stiffness_e(e_in, e_out);
                scatter(cK, e_out, y, have);
            }
        }
        apply_mask(y, masked);
    }

    // y = mask o P^T R_e P x with R_e the Taylor-Galerkin right-hand side
    // kernel: int phi a (e.grad x) + int (e.grad phi) c (e.grad x).
    void advect(const T a, const T c, const T* x, T* y)
    {
        ++applications;
        gather(x);
        t_advect.start();
        bk::tg::TaylorGalerkin<T, nq, nm>(nelmt, B, Dq, w1d, detJ, et[0], et[1], et[2], a, c, e_in, e_out);
        t_advect.stop();
        scatter(T(1), e_out, y, false);
        apply_mask(y, true);
    }
};

} // namespace bp

#endif // BP_OPERATOR_H
