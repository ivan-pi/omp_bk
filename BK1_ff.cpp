// BK1_ff.cpp -- BK1 (mass operator) in float-float arithmetic.
//
// The kernel is the generic sum factorization of bk1_sumfact.h instantiated
// with T = ffloat (float_float.h): a pair of floats carrying ~48 significant
// bits, with every + and * expanded into fp32 adds, multiplies and fused
// multiply-adds.  The point is hardware whose fp64 rate is a small fraction
// of fp32 (consumer GPUs): float-float costs about 18 fp32 flops per
// multiply-add (27 with -DFF_IEEE_ADD), which is cheaper than fp64 at 1/32
// or 1/64 of the fp32 rate, and it reads and writes the same 8 bytes per
// value.
//
// The driver runs the same kernel in double, float and float-float on the
// same data and prints, per precision, the throughput and the relative L2
// error against the double result.  From the repository root:  make BK1_ff
//
// Run:  ./BK1_ff [p=2] [nelmt=524288] [ntests=5]
//   BK_RANDOM=0  constant in = 3.0 and JxW = 1.0 (the BK1 data; the float
//                line then prints BK1's norm).  Default: pseudo-random data
//                with a fixed seed, in uniform on [-1, 1), JxW on [0.5, 1.5).
//
// The error-free transformations behind ffloat need strict IEEE rounding:
// never build with -ffast-math, and on x86 add -mfma or -march=native so
// std::fma is an instruction (the output states whether it is).

#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>

#include "bk_common.h"
#include "float_float.h"
#include "bk1_sumfact.h"

using namespace bk;

namespace {

template <typename T>
const char* precision_name()
{
    if (std::is_same_v<T, double>) {
        return "double";
    }
    if (std::is_same_v<T, float>) {
        return "float";
    }
    return "float-float";
}

// Round (or split) the double-precision test data to the working type.
template <typename T>
std::vector<T> convert(const std::vector<double>& v)
{
    std::vector<T> w(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        w[i] = T(v[i]);
    }
    return w;
}

// ||x - ref|| / ||ref||, accumulated in double.
template <typename T>
double relative_error(const std::vector<T>& x, const std::vector<double>& ref)
{
    double num = 0;
    double den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(x[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

// Run the kernel in precision T; return the result widened to double.  The
// timing is the minimum wall time over ntests repetitions, as in BK1.cpp.
template <typename T, int nq>
std::vector<double> run_precision(
    const std::size_t nelmt, const int ntests,
    const std::vector<double>& basis64,
    const std::vector<double>& JxW64,
    const std::vector<double>& in64,
    const std::vector<double>* reference)
{
    const std::vector<T> basis = convert<T>(basis64);
    const std::vector<T> JxW   = convert<T>(JxW64);
    const std::vector<T> in    = convert<T>(in64);
    std::vector<T>       out(in.size());

    const std::size_t size_inout = in.size();
    const std::size_t size_JxW   = JxW.size();
    [[maybe_unused]] const std::size_t size_basis = basis.size();   // only in the map clause

    const T* d_basis = basis.data();
    const T* d_JxW   = JxW.data();
    const T* d_in    = in.data();
    T*       d_out   = out.data();

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma omp target data \
        map(to: d_basis[:size_basis]) \
        map(to: d_JxW[:size_JxW], d_in[:size_inout]) \
        map(tofrom: d_out[:size_inout])
    for (int t = 0; t < ntests; ++t) {
        auto start = high_resolution_clock::now();
        SumFactorization<T, nq>(nelmt, d_basis, d_JxW, d_in, d_out);
        auto stop = high_resolution_clock::now();
        duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }

    // GDoF/s, and GB/s for read in + write out + read JxW at sizeof(T) bytes
    const double dof_rate  = 1.0e-9 * size_inout / elapsed;
    const double byte_rate = 1.0e-9 * sizeof(T) * (2 * size_inout + size_JxW) / elapsed;

    std::cout << std::left << std::setw(12) << precision_name<T>() << std::right
              << " GDoF/s = " << std::setw(9) << dof_rate
              << " GB/s = "   << std::setw(9) << byte_rate
              << " norm = "   << std::setprecision(10) << norm2(out.data(), out.size())
              << std::setprecision(3);
    if (reference != nullptr) {
        std::cout << " rel. error = " << relative_error(out, *reference);
    }
    std::cout << std::setprecision(6) << "\n";

    std::vector<double> out64(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out64[i] = static_cast<double>(out[i]);
    }
    return out64;
}

template <int nq>
void run_test(const std::size_t nelmt, const int ntests, const bool random_data)
{
    constexpr int nm = nq - 1;

    const std::array<double, nm * nq> b = make_test_basis<double, nm, nq>();
    const std::vector<double> basis(b.begin(), b.end());
    std::vector<double> JxW(nelmt * nq * nq * nq, 1.0);
    std::vector<double> in (nelmt * nm * nm * nm, 3.0);

    if (random_data) {
        std::mt19937 gen(20240601u);
        std::uniform_real_distribution<double> u_in(-1.0, 1.0);
        std::uniform_real_distribution<double> u_JxW(0.5, 1.5);
        for (double& x : in) {
            x = u_in(gen);
        }
        for (double& x : JxW) {
            x = u_JxW(gen);
        }
    }

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " nq = " << nq
              << " data = " << (random_data ? "random" : "constant")
              << " fma = " << (ffloat_hardware_fma ? "hardware" : "library call")
              << "\n";

    const std::vector<double> reference =
        run_precision<double, nq>(nelmt, ntests, basis, JxW, in, nullptr);
    run_precision<float,  nq>(nelmt, ntests, basis, JxW, in, &reference);
    run_precision<ffloat, nq>(nelmt, ntests, basis, JxW, in, &reference);
}

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288, as BK1

} // namespace

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const bool random_data = get_env("BK_RANDOM").value_or("1") != "0";

    // Runtime p -> compile-time nq = p + 2, one instantiation per order.
    switch (p) {
        case 1: run_test< 3>(nelmt, ntests, random_data); break;
        case 2: run_test< 4>(nelmt, ntests, random_data); break;
        case 3: run_test< 5>(nelmt, ntests, random_data); break;
        case 4: run_test< 6>(nelmt, ntests, random_data); break;
        case 5: run_test< 7>(nelmt, ntests, random_data); break;
        case 6: run_test< 8>(nelmt, ntests, random_data); break;
        case 7: run_test< 9>(nelmt, ntests, random_data); break;
        case 8: run_test<10>(nelmt, ntests, random_data); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return 0;
}
