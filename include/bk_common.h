// bk_common.h -- shared by every driver: array views, test data, command
// line, order dispatch, timing and the result line.

#ifndef BK_COMMON_H
#define BK_COMMON_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace bk {

// ndview: multi-dimensional accessor with compile-time extents, row-major
// (last index fastest); constexpr, so usable in device code as is.
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
    for (std::size_t i = 0; i < n; ++i) {
        sumsq += AccumT(v[i]) * AccumT(v[i]);
    }
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
    for (int a = 0; a < rows; ++a) {
        for (int b = 0; b < cols; ++b) {
            m[a * cols + b] = std::cos(T(a * cols + b));
        }
    }
    return m;
}

// The same with run-time extents.
template <typename T>
std::vector<T> make_test_basis(const int rows, const int cols)
{
    std::vector<T> m(std::size_t(rows) * cols);
    for (int a = 0; a < rows; ++a) {
        for (int b = 0; b < cols; ++b) {
            m[a * cols + b] = std::cos(T(a * cols + b));
        }
    }
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
    if (const char* val = std::getenv(name)) {
        return std::string(val);
    }
    return std::nullopt;
}

// n rounded up to a multiple of m
constexpr int round_up(const int n, const int m) { return ((n + m - 1) / m) * m; }

// Integer environment option, `fallback` when unset.
inline int env_int(const char* name, const int fallback)
{
    const auto v = get_env(name);
    return v ? std::atoi(v->c_str()) : fallback;
}

// Command line of the openmp/ drivers: <order> [nelmt] [ntests].
constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // 524288

struct run_args {
    int order;
    std::size_t nelmt;
    int ntests;
};

inline run_args parse_args(const int argc, char** argv, const int default_order)
{
    return {(argc > 1) ? std::atoi(argv[1]) : default_order,
            (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt,
            (argc > 3) ? std::atoi(argv[3]) : 5};
}

// Run-time order -> compile-time order: calls f(std::integral_constant<int,
// order>{}) for an order in [lo, hi] and returns true; otherwise reports
// "unsupported <what> = <order>" and returns false.  A build with
// -DBK_ORDER=n instantiates order n only (the GPU builds: one kernel set per
// binary, so the device compiler sees one order at a time).
template <int lo, int hi, typename F>
bool with_order_impl(const int order, F& f)
{
    if (order == lo) {
        f(std::integral_constant<int, lo>{});
        return true;
    }
    if constexpr (lo < hi) {
        return with_order_impl<lo + 1, hi>(order, f);
    } else {
        return false;
    }
}

template <int lo, int hi, typename F>
bool with_order(const char* what, const int order, F&& f)
{
#ifdef BK_ORDER
    static_assert(lo <= BK_ORDER && BK_ORDER <= hi, "BK_ORDER outside the driver's orders");
    if (with_order_impl<BK_ORDER, BK_ORDER>(order, f)) {
        return true;
    }
    std::cerr << "unsupported " << what << " = " << order << " (this binary: " << BK_ORDER << ")\n";
#else
    if (with_order_impl<lo, hi>(order, f)) {
        return true;
    }
    std::cerr << "unsupported " << what << " = " << order << " (supported: " << lo << ".." << hi << ")\n";
#endif
    return false;
}

// Minimum wall time in seconds of ntests calls of run().
template <typename F>
double time_min(const int ntests, F&& run)
{
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < ntests; ++t) {
        const auto start = std::chrono::steady_clock::now();
        run();
        const auto stop = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(stop - start).count());
    }
    return best;
}

// The result line of the openmp/ drivers: GDoF/s, and GB/s for `bytes`
// moved per run.
inline void print_rates(const std::size_t nelmt, const std::size_t dofs, const std::size_t bytes,
                        const double seconds)
{
    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " GDoF/s = " << 1.0e-9 * dofs / seconds
              << " GB/s = "   << 1.0e-9 * bytes / seconds << "\n";
}

} // namespace bk

#endif // BK_COMMON_H
