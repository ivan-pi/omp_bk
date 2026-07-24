#pragma once

#include <optional>

// ---------------------------------------------------------------------------
// ndview: zero-overhead multi-dimensional accessor, compile-time extents.
// No __host__ __device__ annotations: everything is constexpr and touches
// only template parameters. Under nvcc compile with --expt-relaxed-constexpr;
// clang CUDA / hipcc need no flag; plain C++ (OpenMP, Kokkos host) is
// unaffected.
// ---------------------------------------------------------------------------
template <typename T, int... Extents>
struct ndview {
    static constexpr int rank = sizeof...(Extents);
    static constexpr int size = (Extents * ...);

    T* data;

    template <typename... Is>
    constexpr T& operator()(Is... is) const {
        static_assert(sizeof...(Is) == rank, "index count must match rank");
        int idx = 0;
        ((idx = idx * Extents + static_cast<int>(is)), ...);
        return data[idx];
    }
};


// Euclidean norm: sqrt of the sum of squares. The accumulation is carried
// out in AccumT (double by default) so that float data does not lose
// precision in a long sum; for T == double this is the plain sequential sum.
template <typename AccumT = double, typename T>
AccumT norm2(const T* v, const std::size_t n)
{
    AccumT sumsq = 0;
    for (std::size_t i = 0; i < n; ++i)
        sumsq += AccumT(v[i]) * AccumT(v[i]);
    return std::sqrt(sumsq);
}

template <typename AccumT = double, typename T>
AccumT norm2(const std::vector<T>& v)
{
	return norm2(v.data(), v.size());
}


// Test-harness basis data: B(i, p) = cos(i * nq + p), stored mode-major as
// basis[i * nq + p] -- the layout the kernel's B view expects.
template <typename T, int nq>
std::array<T, (nq - 1) * nq> make_test_basis()
{
    constexpr int nm = nq - 1;
    std::array<T, nm * nq> basis{};
    for (int i = 0; i < nm; i++)
        for (int p = 0; p < nq; p++)
            basis[i * nq + p] = std::cos(T(i * nq + p));
    return basis;
}

// Test-harness basis data: M(a, b) = cos(a * cols + b), stored row-major --
// covers both the (nm x nq) interpolation basis and the (nq x nq) derivative
// matrix used here.
template <typename T, int rows, int cols>
std::array<T, rows * cols> make_test_basis()
{
    std::array<T, rows * cols> m{};
    for (int a = 0; a < rows; a++)
        for (int b = 0; b < cols; b++)
            m[a * cols + b] = std::cos(T(a * cols + b));
    return m;
}

std::optional<std::string> get_env(const char* name)
{
    if (const char* val = std::getenv(name))
        return std::string(val);
    return std::nullopt;
}
