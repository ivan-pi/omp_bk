#ifndef FLOAT_FLOAT_H
#define FLOAT_FLOAT_H

// float_float.h -- "float-float" (double-float) arithmetic.
//
// An ffloat is the unevaluated sum hi + lo of two IEEE binary32 numbers with
// |lo| <= ulp(hi) / 2, which carries about 48 significant bits (2 x 24, a few
// bits short of double's 53).  The exponent range is that of float, so values
// must stay within ~1e-38 .. 3e38 and no denormal arithmetic is guaranteed.
// Every operation is a chain of fp32 adds, multiplies and fused multiply-adds:
// on hardware whose fp64 throughput is a fraction of fp32 (consumer GPUs) this
// is the usual way to get near-double accuracy at near-fp32 cost.
//
// Costs per operation (fp32 flops): * is 7; + and - are 11 by default (the
// QD library's sloppy_add: the rounding error of lo + lo is dropped, which
// costs digits only when a + b cancels to far below |a|) or 20 with
// -DFF_IEEE_ADD (QD's ieee_add: both parts summed error-free).  On BK1 with
// random data both reach the same ~3e-15 relative error against double, and
// the sloppy add runs the kernel about 1.5x faster at p = 8.
//
// The building blocks are the error-free transformations TwoSum (Knuth) and
// TwoProd (Dekker, via fma), which only hold under strict IEEE rounding:
// no -ffast-math / -Ofast, no extended-precision evaluation, and std::fma
// must be a real fused multiply-add for the result to be exact (a libm fmaf
// is still correct, only slow: on x86 build with -mfma or -march=native).

#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(__FAST_MATH__)
#error "float-float arithmetic needs IEEE semantics: do not compile with -ffast-math / -Ofast"
#endif

namespace bk {

static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE binary32");
static_assert(FLT_EVAL_METHOD == 0, "float expressions must be evaluated in float (no x87 excess precision)");

// True when the compiler emits a fused multiply-add instruction for std::fma
// on float (GCC's FP_FAST_FMAF; clang only exposes the ISA macros).
#if defined(FP_FAST_FMAF) || defined(__FMA__) || defined(__ARM_FEATURE_FMA)
constexpr bool ffloat_hardware_fma = true;
#else
constexpr bool ffloat_hardware_fma = false;
#endif

struct ffloat {
    float hi;
    float lo;

    ffloat() = default;                          // trivial: work arrays need no zeroing
    constexpr ffloat(float h, float l) : hi(h), lo(l) {}
    constexpr ffloat(float x) : hi(x), lo(0.0f) {}
    constexpr ffloat(int x) : ffloat(static_cast<double>(x)) {}
    // Exact split of a double into the leading 24 bits and the next 24 bits
    // (for |x| within float's range).
    constexpr ffloat(double x)
        : hi(static_cast<float>(x)),
          lo(static_cast<float>(x - static_cast<double>(static_cast<float>(x)))) {}

    explicit constexpr operator double() const { return static_cast<double>(hi) + static_cast<double>(lo); }
    explicit constexpr operator float() const { return hi; }   // hi == fl(hi + lo) by the invariant
};

static_assert(sizeof(ffloat) == 2 * sizeof(float), "ffloat is two packed floats");
static_assert(std::is_trivially_copyable_v<ffloat>, "ffloat must be bitwise copyable (OpenMP map)");
static_assert(std::is_trivially_default_constructible_v<ffloat>, "ffloat arrays must not be zero-filled");

namespace ff {

// (s, e) with s + e == a + b exactly, s = fl(a + b).  6 flops.
inline ffloat two_sum(float a, float b)
{
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return ffloat{s, e};
}

// Same result, 3 flops, valid only when |a| >= |b| or a == 0.
inline ffloat quick_two_sum(float a, float b)
{
    const float s = a + b;
    const float e = b - (s - a);
    return ffloat{s, e};
}

// (p, e) with p + e == a * b exactly, p = fl(a * b).  1 multiply + 1 fma.
inline ffloat two_prod(float a, float b)
{
    const float p = a * b;
    const float e = std::fma(a, b, -p);
    return ffloat{p, e};
}

} // namespace ff

inline ffloat operator-(ffloat a) { return ffloat{-a.hi, -a.lo}; }

#if defined(FF_IEEE_ADD)
// Accurate addition (QD's ieee_add): both the hi and the lo parts are summed
// with their rounding errors, then the pair is renormalised twice.  20 flops.
inline ffloat operator+(ffloat a, ffloat b)
{
    ffloat s = ff::two_sum(a.hi, b.hi);
    const ffloat t = ff::two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = ff::quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return ff::quick_two_sum(s.hi, s.lo);
}
#else
// Fast addition (QD's sloppy_add): the hi parts are summed error-free, the lo
// parts in plain fp32, then the pair is renormalised once.  11 flops.
inline ffloat operator+(ffloat a, ffloat b)
{
    ffloat s = ff::two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return ff::quick_two_sum(s.hi, s.lo);
}
#endif

inline ffloat operator-(ffloat a, ffloat b) { return a + (-b); }

// a.hi * b.hi exactly, plus the two cross terms; a.lo * b.lo is below the
// precision of the result and is dropped.
inline ffloat operator*(ffloat a, ffloat b)
{
    ffloat p = ff::two_prod(a.hi, b.hi);
    p.lo = std::fma(a.hi, b.lo, std::fma(a.lo, b.hi, p.lo));
    return ff::quick_two_sum(p.hi, p.lo);
}

inline ffloat& operator+=(ffloat& a, ffloat b)
{
    a = a + b;
    return a;
}

inline ffloat& operator-=(ffloat& a, ffloat b)
{
    a = a - b;
    return a;
}

inline ffloat& operator*=(ffloat& a, ffloat b)
{
    a = a * b;
    return a;
}

} // namespace bk

#endif // FLOAT_FLOAT_H
