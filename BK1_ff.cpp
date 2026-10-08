// BK1_ff.cpp -- BK1 (mass operator) in float-float arithmetic.
//
// The kernel is the generic sum factorization of bk1_sumfact.h instantiated
// with T = ffloat (float_float.h): a pair of floats carrying ~48 significant
// bits, with every + and * expanded into fp32 adds, multiplies and fused
// multiply-adds.  The point is hardware whose fp64 rate is a small fraction
// of fp32 (consumer GPUs): float-float costs about 18 fp32 flops per
// multiply-add, which is cheaper than fp64 at 1/32 or 1/64 of the fp32
// rate, and it reads and writes the same 8 bytes per value.
//
// The driver runs the same kernel in double, float and float-float on the
// same data and prints one row per precision (throughput and norm), then the
// relative L2 error of float and of float-float against the double result.
// Norms and errors are reduced in native double after widening every value.
// The last line is the check: the float-float error must stay below the
// tolerance, else it reads FAIL and the exit status is 1.
// From the repository root:  make BK1_ff
//
// Run:  ./BK1_ff [p=2] [nelmt=524288] [ntests=5]
//   BK_RANDOM=1  pseudo-random in/JxW (bk_common.h) instead of the constant
//                3.0/1.0 of BK1; with the constant data the float row prints
//                BK1's norm.
//   BK_TOL=t     tolerance of the check (default 1e-10: random data gives
//                ~3e-15, the cancelling constant data at p = 1 ~4e-13, and a
//                broken ffloat operation 1e-7 or worse).

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <string>
#include <type_traits>

#include "bk_common.h"
#include "float_float.h"
#include "bk1_sumfact.h"

using namespace bk;

namespace {

// The double test data as seen by precision T: a reference for T = double,
// otherwise a rounded (float) or split (ffloat) copy held in `storage`.
template <typename T>
const std::vector<T>& as_precision(const std::vector<double>& v, std::vector<T>& storage)
{
    if constexpr (std::is_same_v<T, double>) {
        return v;
    } else {
        storage.assign(v.begin(), v.end());
        return storage;
    }
}

// ||x - ref|| / ||ref||, every x[i] widened to double (hi + lo for ffloat),
// both sums accumulated in double.
template <typename T>
double relative_error(const std::vector<T>& x, const std::vector<double>& ref)
{
    assert(x.size() == ref.size());
    double num = 0;
    double den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(x[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

// Time the kernel in precision T, print its row and return its output.
template <typename T, int nq>
std::vector<T> run_precision(
    const char* label, const std::size_t nelmt, const int ntests,
    const std::vector<double>& basis64,
    const std::vector<double>& JxW64,
    const std::vector<double>& in64)
{
    std::vector<T> basis_T;
    std::vector<T> JxW_T;
    std::vector<T> in_T;
    const std::vector<T>& basis = as_precision<T>(basis64, basis_T);
    const std::vector<T>& JxW   = as_precision<T>(JxW64, JxW_T);
    const std::vector<T>& in    = as_precision<T>(in64, in_T);
    std::vector<T> out(in.size());

    const double elapsed = time_sumfact<T, nq>(
        nelmt, ntests, basis.data(), JxW.data(), in.data(), out.data());

    // GDoF/s, and GB/s for read in + write out + read JxW at sizeof(T) bytes
    const double dof_rate  = 1.0e-9 * in.size() / elapsed;
    const double byte_rate = 1.0e-9 * sizeof(T) * (2 * in.size() + JxW.size()) / elapsed;

    std::cout << std::left << std::setw(12) << label << std::right
              << " GDoF/s = " << std::setw(9) << dof_rate
              << " GB/s = "   << std::setw(9) << byte_rate
              << " norm = "   << std::setprecision(10) << norm2(out) << std::setprecision(6)
              << "\n";
    return out;
}

// Returns true when the float-float result is within tol of the double one.
template <int nq>
bool run_test(const std::size_t nelmt, const int ntests, const bool random_data,
              const double tol)
{
    constexpr int nm = nq - 1;

    const std::array<double, nm * nq> b = make_test_basis<double, nm, nq>();
    const std::vector<double> basis(b.begin(), b.end());
    std::vector<double> JxW(nelmt * nq * nq * nq, 1.0);
    std::vector<double> in (nelmt * nm * nm * nm, 3.0);
    if (random_data) {
        fill_random_test_data(in, JxW);
    }

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " nq = " << nq
              << " data = " << (random_data ? "random" : "constant")
              << " fma = " << (ffloat_hardware_fma ? "hardware" : "library call")
              << "\n";

    const std::vector<double> ref =
        run_precision<double, nq>("double", nelmt, ntests, basis, JxW, in);
    const double err_float =
        relative_error(run_precision<float, nq>("float", nelmt, ntests, basis, JxW, in), ref);
    const double err_ff =
        relative_error(run_precision<ffloat, nq>("float-float", nelmt, ntests, basis, JxW, in), ref);

    const bool ok = err_ff <= tol;
    std::cout << "float        rel. error vs double = " << err_float << "\n"
              << (ok ? "ok" : "FAIL")
              << ": float-float rel. error vs double = " << err_ff
              << " (tol = " << tol << ")\n";
    return ok;
}

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288, as BK1

} // namespace

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const bool random_data = get_env("BK_RANDOM").has_value();
    const double tol = std::atof(get_env("BK_TOL").value_or("1e-10").c_str());

    // Runtime p -> compile-time nq = p + 2, one instantiation per order.
    bool ok = false;
    switch (p) {
        case 1: ok = run_test< 3>(nelmt, ntests, random_data, tol); break;
        case 2: ok = run_test< 4>(nelmt, ntests, random_data, tol); break;
        case 3: ok = run_test< 5>(nelmt, ntests, random_data, tol); break;
        case 4: ok = run_test< 6>(nelmt, ntests, random_data, tol); break;
        case 5: ok = run_test< 7>(nelmt, ntests, random_data, tol); break;
        case 6: ok = run_test< 8>(nelmt, ntests, random_data, tol); break;
        case 7: ok = run_test< 9>(nelmt, ntests, random_data, tol); break;
        case 8: ok = run_test<10>(nelmt, ntests, random_data, tol); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return ok ? 0 : 1;
}
