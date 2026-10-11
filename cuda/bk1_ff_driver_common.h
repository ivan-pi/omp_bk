#ifndef BK1_FF_DRIVER_COMMON_H
#define BK1_FF_DRIVER_COMMON_H

// bk1_ff_driver_common.h -- what the GPU drivers of the float-float kernel
// share: the command line and environment of BK1_ff, the test data with
// its double reference computed on the host (bk1_sumfact.h), the dispatch
// from the runtime order p to the compile-time nq, and the report of one
// timed run with the ok/FAIL check.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "bk_common.h"
#include "bk1_sumfact.h"

namespace bk {
namespace ffdrv {

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288, as BK1

// Kernel time of the untimed warm-up runs before the timed ones: enough
// for the GPU to leave the idle clock it drops to while the host computes
// the reference (BK_WARMUP=s overrides).
inline double warmup_seconds_default() { return 0.2; }
inline const double warmup_seconds = [] {
    const char* env = std::getenv("BK_WARMUP");
    return (env != nullptr) ? std::atof(env) : warmup_seconds_default();
}();

// ./driver [p=2] [nelmt=524288] [ntests=5] [nelmtPerBatch=0: the default
// of mass_operator_ff_traits]; BK_RANDOM=1 and BK_TOL=t as in BK1_ff,
// BK_WARMUP=s the warm-up kernel time in seconds (default 0.2).
struct Args {
    int p = 2;
    std::size_t nelmt = default_nelmt;
    int ntests = 5;
    unsigned int batch = 0;
    bool random_data = false;
    double tol = 1e-10;
};

inline Args parse_args(const int argc, char** argv)
{
    Args a;
    if (argc > 1) {
        a.p = std::atoi(argv[1]);
    }
    if (argc > 2) {
        a.nelmt = static_cast<std::size_t>(std::atoll(argv[2]));
    }
    if (argc > 3) {
        a.ntests = std::atoi(argv[3]);
    }
    if (argc > 4) {
        a.batch = static_cast<unsigned int>(std::atoi(argv[4]));
    }
    a.random_data = get_env("BK_RANDOM").has_value();
    a.tol = std::atof(get_env("BK_TOL").value_or("1e-10").c_str());
    return a;
}

// The test data of BK1_ff for one order and its double reference.
template <int nq>
struct TestData {
    static constexpr int nm = nq - 1;
    std::vector<double> basis;
    std::vector<double> JxW;
    std::vector<double> in;
    std::vector<double> ref;

    TestData(const std::size_t nelmt, const bool random_data)
        : JxW(nelmt * nq * nq * nq, 1.0), in(nelmt * nm * nm * nm, 3.0), ref(nelmt * nm * nm * nm)
    {
        const std::array<double, nm * nq> b = make_test_basis<double, nm, nq>();
        basis.assign(b.begin(), b.end());
        if (random_data) {
            fill_random_test_data(in, JxW);
        }
        SumFactorization<double, nq>(nelmt, basis.data(), JxW.data(), in.data(), ref.data());
    }
};

// ||x - ref|| / ||ref|| in double.
inline double relative_error(const std::vector<double>& x, const std::vector<double>& ref)
{
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = x[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

// The header line of a run.
inline void print_header(const std::size_t nelmt, const int nq, const bool random_data,
                         const std::string& device, const unsigned int batch, const int block,
                         const int grid, const std::size_t smem_bytes)
{
    std::cout << "SumFactorization -> nelmt = " << nelmt << " nq = " << nq
              << " data = " << (random_data ? "random" : "constant")
              << " device = " << device
              << " batch = " << batch << " block = " << block
              << " grid = " << grid << " smem = " << smem_bytes << " B\n";
}

// The row of a timed run (elapsed seconds of the fastest of ntests) and the
// check; returns ok.
inline bool report(const char* label, const double elapsed, const std::vector<double>& out,
                   const std::size_t size_JxW, const std::vector<double>& ref, const double tol)
{
    const double dof_rate  = 1.0e-9 * static_cast<double>(out.size()) / elapsed;
    const double byte_rate = 1.0e-9 * sizeof(double) * static_cast<double>(2 * out.size() + size_JxW) / elapsed;
    const double err = relative_error(out, ref);
    const bool ok = err <= tol;
    std::cout << std::left << std::setw(12) << label << std::right
              << " GDoF/s = " << std::setw(9) << dof_rate
              << " GB/s = "   << std::setw(9) << byte_rate
              << " norm = "   << std::setprecision(10) << norm2(out) << std::setprecision(6) << "\n"
              << (ok ? "ok" : "FAIL") << ": " << label << " rel. error vs double = " << err
              << " (tol = " << tol << ")\n";
    return ok;
}

// Runtime p -> compile-time nq = p + 2: run(std::integral_constant<int, nq>)
// for p = 1..8, else false after a message.
template <typename Run>
bool dispatch(const int p, Run&& run)
{
    switch (p) {
        case 1: return run(std::integral_constant<int,  3>{});
        case 2: return run(std::integral_constant<int,  4>{});
        case 3: return run(std::integral_constant<int,  5>{});
        case 4: return run(std::integral_constant<int,  6>{});
        case 5: return run(std::integral_constant<int,  7>{});
        case 6: return run(std::integral_constant<int,  8>{});
        case 7: return run(std::integral_constant<int,  9>{});
        case 8: return run(std::integral_constant<int, 10>{});
        default:
            std::cerr << "unsupported polynomial order p = " << p << " (supported: 1..8)\n";
            return false;
    }
}

} // namespace ffdrv
} // namespace bk

#endif // BK1_FF_DRIVER_COMMON_H
