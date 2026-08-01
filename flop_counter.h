#ifndef FLOP_COUNTER_H
#define FLOP_COUNTER_H

#include <cstddef>

namespace bk {

// ---------------------------------------------------------------------------
// flop_counter<Real>: an operation-counting stand-in for a floating scalar.
//
// Substitute it for the kernel's value type T and every arithmetic operator
// the kernel executes bumps a global tally. Because the loop trip counts of
// the BK kernels are data-independent, running a kernel once (nelmt = 1,
// single-threaded) yields the exact per-element flop count -- no timing, no
// hardware counters, no disassembly.
//
// Convention: one multiply and one add are each counted as one flop, so the
// fused pattern
//     acc += a * b
// contributes 1 multiply + 1 add = 2 flops. This is the usual finite-element
// / roofline convention: a hardware FMA is a single instruction but is still
// counted as two flops. (adds and muls are tallied separately, so a
// count-FMA-as-one convention can be recovered as max(adds, muls) per FMA if
// ever needed.)
//
// The counters are static members, shared by every flop_counter<Real> value,
// so call reset() immediately before the kernel invocation you want to time
// and read them immediately after. Not thread-safe by design -- measure a
// single element serially.
// ---------------------------------------------------------------------------
template <typename Real>
struct flop_counter {
    Real v;

    static inline std::size_t adds = 0;
    static inline std::size_t muls = 0;

    static void        reset() { adds = 0; muls = 0; }
    static std::size_t flops() { return adds + muls; }

    flop_counter()      : v(Real(0)) {}
    flop_counter(Real x) : v(x)      {}   // also matches int 0 via int->Real

    friend flop_counter operator+(flop_counter a, flop_counter b) { ++adds; return flop_counter(a.v + b.v); }
    friend flop_counter operator-(flop_counter a, flop_counter b) { ++adds; return flop_counter(a.v - b.v); }
    friend flop_counter operator*(flop_counter a, flop_counter b) { ++muls; return flop_counter(a.v * b.v); }

    flop_counter& operator+=(flop_counter b) { ++adds; v += b.v; return *this; }
    flop_counter& operator-=(flop_counter b) { ++adds; v -= b.v; return *this; }
    flop_counter& operator*=(flop_counter b) { ++muls; v *= b.v; return *this; }
};

} // namespace bk

#endif // FLOP_COUNTER_H
