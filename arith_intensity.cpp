// ---------------------------------------------------------------------------
// arith_intensity: exact flop / byte / arithmetic-intensity report for the
// BK kernels.
//
// Two independent methods are combined and cross-checked:
//
//   1. Measurement. Each kernel is instantiated with flop_counter<double> in
//      place of the value type and run once (nelmt = 1). Every arithmetic
//      operator the *actual kernel source* performs bumps a counter, so the
//      reported flops are exact -- there is no modelling of the loop bodies.
//
//   2. Closed form. Analytic flop formulas (derived per step, see below) are
//      evaluated and asserted equal to the measured counts. If they ever
//      disagree the program exits non-zero, so this doubles as a regression
//      test on the formulas.
//
// Bytes counted are the streaming DRAM traffic per element, matching the
// "effective GB/s" the timing drivers report: read `in`, write `out`, read
// the per-point geometric factors (JxW for BK1, the 6-component G for BK3 /
// BK5). The small basis / derivative matrices are reused across all elements
// and cached, so -- like the drivers -- they are excluded. Scratch space is
// register / local memory, not DRAM traffic, and is likewise excluded.
//
// Arithmetic intensity = flops / bytes. Bytes use sizeof(float) = 4, the type
// the kernels are actually instantiated with; using double doubles the bytes
// and so halves the intensity (flops are unchanged).
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "flop_counter.h"
#include "bk1_kernel.h"
#include "bk3_kernel.h"
#include "bk5_kernel.h"

namespace {

using Real = double;             // wrapped value type -- values are irrelevant
using C    = bk::flop_counter<Real>;

// Bytes per scalar in the deployed kernels (float). Change to 8 for double.
constexpr long long BYTES_PER_SCALAR = sizeof(float);

bool g_ok = true;               // cleared on any formula/measurement mismatch

struct Counts { long long adds, muls; };

constexpr long long ipow(long long b, int e)
{
    long long r = 1;
    for (int i = 0; i < e; ++i) r *= b;
    return r;
}

// -------- closed-form flop formulas (adds, muls) per element ----------------
// A contraction step `acc = 0; for n<L: acc += a*b` over S outputs contributes
// L*S adds and L*S muls. Summing the per-step contributions:

constexpr Counts formula_bk1(int nq)
{
    const long long m = nq - 1, q = nq;
    const long long shared = 2*q*ipow(m,3) + 2*q*q*m*m + 2*m*ipow(q,3);
    return { shared,                 // adds
             shared + ipow(q,3) };   // muls: extra nq^3 from the JxW scaling
}

constexpr Counts formula_bk3(int nq)
{
    const long long m = nq - 1, q = nq;
    const long long interp = 2*q*ipow(m,3) + 2*q*q*m*m + 2*m*ipow(q,3);
    return { interp + 6*ipow(q,4) + 6*ipow(q,3),    // adds
             interp + 6*ipow(q,4) + 9*ipow(q,3) };  // muls
}

constexpr Counts formula_bk5(int nq)
{
    const long long q = nq;
    return { 6*ipow(q,4) + 6*ipow(q,3),      // adds
             6*ipow(q,4) + 9*ipow(q,3) };    // muls
}

// ----------------------------- measurement ----------------------------------
// Fill helper: values do not affect the (data-independent) trip counts.
std::vector<C> box(std::size_t n) { return std::vector<C>(n, C(Real(1))); }

template <int nq>
Counts measure_bk1()
{
    constexpr int nm = nq - 1;
    auto basis = box(nm*nq), JxW = box(nq*nq*nq),
         in    = box(nm*nm*nm), out = box(nm*nm*nm);
    C::reset();
    bk::bk1::SumFactorization<C, nq>(1, basis.data(), JxW.data(),
                                     in.data(), out.data());
    return { (long long)C::adds, (long long)C::muls };
}

template <int nq>
Counts measure_bk3()
{
    constexpr int nm = nq - 1;
    auto basis = box(nm*nq), dbasis = box(nq*nq), G = box(6*nq*nq*nq),
         in    = box(nm*nm*nm), out = box(nm*nm*nm);
    C::reset();
    bk::bk3::SumFactorization<C, nq>(1, basis.data(), dbasis.data(),
                                     G.data(), in.data(), out.data());
    return { (long long)C::adds, (long long)C::muls };
}

template <int nq>
Counts measure_bk5()
{
    auto dbasis = box(nq*nq), G = box(6*nq*nq*nq),
         in     = box(nq*nq*nq), out = box(nq*nq*nq);
    C::reset();
    bk::bk5::SumFactorization<C, nq>(1, dbasis.data(), G.data(),
                                     in.data(), out.data());
    return { (long long)C::adds, (long long)C::muls };
}

// ------------------------------- reporting ----------------------------------
void print_header()
{
    std::cout << std::right
              << std::setw(4)  << "nq"    << std::setw(4)  << "p"
              << std::setw(12) << "adds"  << std::setw(12) << "muls"
              << std::setw(12) << "flops" << std::setw(9)  << "bytes"
              << std::setw(11) << "flop/DoF"
              << std::setw(12) << "flop/byte"
              << std::setw(12) << "byte/flop" << "\n";
}

void print_row(int nq, int p, Counts meas, Counts pred,
               long long bytes, long long dofs)
{
    if (meas.adds != pred.adds || meas.muls != pred.muls) {
        g_ok = false;
        std::cout << "  [MISMATCH nq=" << nq
                  << " measured(add=" << meas.adds << ",mul=" << meas.muls
                  << ") formula(add=" << pred.adds << ",mul=" << pred.muls
                  << ")]\n";
    }
    const long long flops = meas.adds + meas.muls;
    std::cout << std::right << std::fixed << std::setprecision(3)
              << std::setw(4)  << nq
              << std::setw(4)  << (p >= 0 ? std::to_string(p) : std::string("-"))
              << std::setw(12) << meas.adds
              << std::setw(12) << meas.muls
              << std::setw(12) << flops
              << std::setw(9)  << bytes
              << std::setw(11) << double(flops) / double(dofs)
              << std::setw(12) << double(flops) / double(bytes)
              << std::setw(12) << double(bytes) / double(flops) << "\n";
}

template <typename F, int... Ns>
void for_orders(F f, std::integer_sequence<int, Ns...>)
{
    (f(std::integral_constant<int, Ns>{}), ...);
}

} // namespace

int main()
{
    std::cout << "Per-element arithmetic intensity of the BK kernels\n"
              << "(bytes = streaming DRAM traffic, sizeof(float) = "
              << BYTES_PER_SCALAR << "; a*b counted as 2 flops)\n";

    std::cout << "\nBK1  (mass matrix; DoF = nm^3, bytes = 2*nm^3 + nq^3)\n";
    print_header();
    for_orders([](auto ic) {
        constexpr int nq = ic.value, nm = nq - 1;
        const Counts meas = measure_bk1<nq>();
        const long long bytes = BYTES_PER_SCALAR * (2*ipow(nm,3) + ipow(nq,3));
        print_row(nq, nq - 2, meas, formula_bk1(nq), bytes, ipow(nm,3));
    }, std::integer_sequence<int, 3,4,5,6,7,8,9,10>{});

    std::cout << "\nBK3  (Poisson; DoF = nm^3, bytes = 2*nm^3 + 6*nq^3)\n";
    print_header();
    for_orders([](auto ic) {
        constexpr int nq = ic.value, nm = nq - 1;
        const Counts meas = measure_bk3<nq>();
        const long long bytes = BYTES_PER_SCALAR * (2*ipow(nm,3) + 6*ipow(nq,3));
        print_row(nq, nq - 2, meas, formula_bk3(nq), bytes, ipow(nm,3));
    }, std::integer_sequence<int, 3,4,5,6,7,8,9,10>{});

    std::cout << "\nBK5  (collocated Laplacian; DoF = nq^3, bytes = 8*nq^3)\n";
    print_header();
    for_orders([](auto ic) {
        constexpr int nq = ic.value;
        const Counts meas = measure_bk5<nq>();
        const long long bytes = BYTES_PER_SCALAR * (8*ipow(nq,3));
        print_row(nq, -1, meas, formula_bk5(nq), bytes, ipow(nq,3));
    }, std::integer_sequence<int, 2,3,4,5,6,7,8>{});

    std::cout << "\n" << (g_ok
        ? "All measured counts match the closed-form formulas.\n"
        : "WARNING: measured counts disagree with the formulas (see above).\n");
    return g_ok ? 0 : 1;
}
