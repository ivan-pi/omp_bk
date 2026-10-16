#ifndef BK_COMMON_H
#define BK_COMMON_H

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <vector>

namespace bk {

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

// Pseudo-random test data (BK_RANDOM=1 in the drivers): a deterministic LCG
// fills `in` with values in [-1, 1] and `JxW` with 1.5 + [-1, 1].  Every
// draw is a 24-bit integer scaled by a power of two, so float and double
// receive bit-identical values.
template <typename T>
void fill_random_test_data(std::vector<T>& in, std::vector<T>& JxW,
                           uint32_t seed = 12345u)
{
    uint32_t s = seed;
    auto next = [&] {
        s = 1664525u * s + 1013904223u;
        return T(s >> 8) / T(1 << 23) - T(1);
    };
    for (auto& v : in) {
        v = next();
    }
    for (auto& v : JxW) {
        v = T(1.5) + next();
    }
}

inline std::optional<std::string> get_env(const char* name)
{
    if (const char* val = std::getenv(name))
        return std::string(val);
    return std::nullopt;
}

// Machine-readable results: when BK_CSV names a file, every driver appends
// one row per kernel and run to it (a header first when the file is new):
//   driver,kernel,nq,nelmt,batch,gdofs,gbs,norm,error,ok
// so that a sweep is a shell loop and no output has to be parsed.  batch is
// the elements per team or block (0 where the kernel has none), error the
// relative L2 error against the double result (empty for double itself).
inline void append_csv_row(const char* driver, const std::string& kernel, const int nq,
                           const std::size_t nelmt, const int batch, const double gdofs,
                           const double gbs, const double norm, const std::optional<double> error,
                           const bool ok)
{
    const std::optional<std::string> file = get_env("BK_CSV");
    if (!file) {
        return;
    }
    bool is_new = true;
    {
        std::ifstream probe(*file);
        is_new = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream out(*file, std::ios::app);
    if (is_new) {
        out << "driver,kernel,nq,nelmt,batch,gdofs,gbs,norm,error,ok\n";
    }
    out << driver << ',' << kernel << ',' << nq << ',' << nelmt << ',' << batch << ','
        << std::setprecision(6) << gdofs << ',' << gbs << ','
        << std::setprecision(17) << norm << ',';
    if (error) {
        out << std::setprecision(6) << *error;
    }
    out << ',' << (ok ? "ok" : "FAIL") << '\n';
}

} // namespace bk

#endif // BK_COMMON_H
