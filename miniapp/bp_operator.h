#ifndef BP_OPERATOR_H
#define BP_OPERATOR_H

// The assembled operator  A = cM M + cK K  applied matrix-free:
//
//     y = mask o P^T ( cM M_e + cK K_e ) P x
//
// with M_e / K_e the element mass / stiffness kernels (BK1 / BK3 for the
// Gauss-Legendre family, lumped GLL mass / BK5 for the collocated family)
// and `mask` zeroing the Dirichlet boundary nodes.  The mass solve (cM = 1,
// cK = 0, no mask), the Poisson solve (cM = 0, cK = 1) and the implicit
// diffusion step (cM = 1, cK = theta dt) are all instances of this one
// operator.

#include <cassert>
#include <chrono>
#include <cstddef>

#include "../bk1_kernel.h"
#include "../bk3_kernel.h"
#include "../bk5_kernel.h"
#include "bp_backend.h"

namespace bp {

struct Stopwatch {
    using clock = std::chrono::steady_clock;
    double seconds = 0.0;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    void stop() { seconds += std::chrono::duration<double>(clock::now() - t0).count(); }
};

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

    // basis and metric (bp_basis.h, bp_geometry.h) in the kernels' layouts
    const T* B = nullptr;      // BK1/BK3 basis (nm x nq); unused when collocated
    const T* D = nullptr;      // BK3: Dq (n, p); BK5: Dq^T (p, n)
    const T* JxW = nullptr;
    const T* G = nullptr;
    const T* mask = nullptr;   // nullptr: no Dirichlet condition

    // E-vector work space
    T* e_in = nullptr;
    T* e_mass = nullptr;
    T* e_stiff = nullptr;

    Stopwatch t_restrict;
    Stopwatch t_mass;
    Stopwatch t_stiff;
    Stopwatch t_prolong;
    Stopwatch t_combine;
    long applications = 0;

    double seconds() const
    {
        return t_restrict.seconds + t_mass.seconds + t_stiff.seconds
             + t_prolong.seconds + t_combine.seconds;
    }

    // out = M_e in
    void mass(const T* in, T* out)
    {
        if constexpr (collocated) {
            backend::pointwise(nE, JxW, in, out);
        } else {
            bk::bk1::SumFactorization<T, nq>(nelmt, B, JxW, in, out);
        }
    }

    // out = K_e in
    void stiffness(const T* in, T* out)
    {
        if constexpr (collocated) {
            bk::bk5::SumFactorization<T, nq>(nelmt, D, G, in, out);
        } else {
            bk::bk3::SumFactorization<T, nq>(nelmt, B, D, G, in, out);
        }
    }

    // y = mask o P^T (cM M_e + cK K_e) P x
    void apply(const T cM, const T cK, const T* x, T* y)
    {
        assert(nE == nelmt * nm3);
        ++applications;

        t_restrict.start();
        backend::restrict_(nL, nE, e_to_l, x, e_in);
        t_restrict.stop();

        T* e_out = e_in;
        if (cM != T(0) && cK != T(0)) {
            t_mass.start();
            mass(e_in, e_mass);
            t_mass.stop();
            t_stiff.start();
            stiffness(e_in, e_stiff);
            t_stiff.stop();
            t_combine.start();
            backend::axpby(nE, cM, e_mass, cK, e_stiff);
            t_combine.stop();
            e_out = e_stiff;
        } else if (cM != T(0)) {
            t_mass.start();
            mass(e_in, e_mass);
            t_mass.stop();
            if (cM != T(1)) {
                t_combine.start();
                backend::scale(nE, cM, e_mass);
                t_combine.stop();
            }
            e_out = e_mass;
        } else if (cK != T(0)) {
            t_stiff.start();
            stiffness(e_in, e_stiff);
            t_stiff.stop();
            if (cK != T(1)) {
                t_combine.start();
                backend::scale(nE, cK, e_stiff);
                t_combine.stop();
            }
            e_out = e_stiff;
        } else {
            backend::fill(nE, T(0), e_in);
        }

        t_prolong.start();
        if (atomic_scatter) {
            backend::prolong_atomic(nL, nE, e_to_l, e_out, y);
        } else {
            backend::prolong(nL, nE, l_offsets, l_to_e, e_out, y);
        }
        if (mask != nullptr) {
            backend::pointwise_inplace(nL, mask, y);
        }
        t_prolong.stop();
    }
};

} // namespace bp

#endif // BP_OPERATOR_H
