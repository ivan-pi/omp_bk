#ifndef BP_BACKEND_H
#define BP_BACKEND_H

// The programming-model-specific layer of the mini-app, here OpenMP target
// offload.  Together with the element kernels (bk1/bk3/bk5_kernel.h) these
// are the only functions that run on the device; a port to OpenACC, Kokkos
// or CUDA replaces this file and the kernels and keeps everything else.
//
// All pointers are host pointers that the caller has mapped in an enclosing
// `target data` region, so the map clauses below only assert presence and
// never move data (the same convention as the kernels).  Every function is
// synchronous, which keeps the phase timers in bp_operator.h honest.

#include <cstddef>

namespace bp {
namespace backend {

// E = P L
template <typename T>
void restrict_(const std::size_t nL, const std::size_t nE, const int* __restrict__ e_to_l,
               const T* __restrict__ L, T* __restrict__ E)
{
    #pragma omp target teams loop map(to: e_to_l[:nE], L[:nL]) map(from: E[:nE])
    for (std::size_t s = 0; s < nE; ++s) {
        E[s] = L[e_to_l[s]];
    }
}

// L = P^T E, deterministic: each L entry sums its own E entries, so the
// result is bitwise reproducible from run to run.
template <typename T>
void prolong(const std::size_t nL, const std::size_t nE, const int* __restrict__ l_offsets,
             const int* __restrict__ l_to_e, const T* __restrict__ E,
             T* __restrict__ L)
{
    #pragma omp target teams loop map(to: l_offsets[:nL + 1], l_to_e[:nE], E[:nE]) map(from: L[:nL])
    for (std::size_t g = 0; g < nL; ++g) {
        T s = 0;
        for (int t = l_offsets[g]; t < l_offsets[g + 1]; ++t) {
            s += E[l_to_e[t]];
        }
        L[g] = s;
    }
}

// L = P^T E with atomic adds: the direct transpose of restrict_, no
// transpose map needed, but the summation order varies between runs.
template <typename T>
void prolong_atomic(const std::size_t nL, const std::size_t nE,
                    const int* __restrict__ e_to_l, const T* __restrict__ E,
                    T* __restrict__ L)
{
    #pragma omp target teams loop map(from: L[:nL])
    for (std::size_t g = 0; g < nL; ++g) {
        L[g] = 0;
    }
    // `atomic` may not nest inside a `loop` region, hence the explicit form
    #pragma omp target teams distribute parallel for map(to: e_to_l[:nE], E[:nE]) map(tofrom: L[:nL])
    for (std::size_t s = 0; s < nE; ++s) {
        #pragma omp atomic update
        L[e_to_l[s]] += E[s];
    }
}

// y = x
template <typename T>
void copy(const std::size_t n, const T* __restrict__ x, T* __restrict__ y)
{
    #pragma omp target teams loop map(to: x[:n]) map(from: y[:n])
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

// x = a
template <typename T>
void fill(const std::size_t n, const T a, T* __restrict__ x)
{
    #pragma omp target teams loop map(from: x[:n])
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = a;
    }
}

// x = a x
template <typename T>
void scale(const std::size_t n, const T a, T* __restrict__ x)
{
    #pragma omp target teams loop map(tofrom: x[:n])
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = a * x[i];
    }
}

// y = a x + b y
template <typename T>
void axpby(const std::size_t n, const T a, const T* __restrict__ x, const T b, T* __restrict__ y)
{
    #pragma omp target teams loop map(to: x[:n]) map(tofrom: y[:n])
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + b * y[i];
    }
}

// z = a x + b y
template <typename T>
void lincomb(const std::size_t n, const T a, const T* __restrict__ x,
             const T b, const T* __restrict__ y, T* __restrict__ z)
{
    #pragma omp target teams loop map(to: x[:n], y[:n]) map(from: z[:n])
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = a * x[i] + b * y[i];
    }
}

// y = d o x (pointwise product: Jacobi preconditioner, Dirichlet mask,
// collocated mass matrix)
template <typename T>
void pointwise(const std::size_t n, const T* __restrict__ d, const T* __restrict__ x, T* __restrict__ y)
{
    #pragma omp target teams loop map(to: d[:n], x[:n]) map(from: y[:n])
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = d[i] * x[i];
    }
}

// y = d o y
template <typename T>
void pointwise_inplace(const std::size_t n, const T* __restrict__ d, T* __restrict__ y)
{
    #pragma omp target teams loop map(to: d[:n]) map(tofrom: y[:n])
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = d[i] * y[i];
    }
}

// x . y, accumulated in the working precision on the device
template <typename T>
T dot(const std::size_t n, const T* __restrict__ x, const T* __restrict__ y)
{
    T s = 0;
    #pragma omp target teams loop reduction(+: s) map(to: x[:n], y[:n])
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

// Host <-> device transfers of an array that the enclosing data region holds.
template <typename T>
void to_device(const std::size_t n, T* x)
{
    #pragma omp target update to(x[:n])
}

template <typename T>
void from_device(const std::size_t n, T* x)
{
    #pragma omp target update from(x[:n])
}

} // namespace backend
} // namespace bp

#endif // BP_BACKEND_H
