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
// Costs in fp32 flops: * is 7, + and - are 11, so a multiply-add is 18 (one
// fp64 FMA does the same work on hardware with full-rate fp64).  The
// addition is the QD library's sloppy_add: the rounding error of lo + lo is
// dropped, which costs digits only when a + b cancels to far below |a|.  On
// BK1 it reaches the same ~3e-15 relative error against double as QD's
// 20-flop ieee_add and runs the kernel about 1.5x faster at p = 8.
//
// The building blocks are the error-free transformations TwoSum (Knuth) and
// TwoProd (Dekker, via fma), which only hold under strict IEEE rounding: no
// -ffast-math / -Ofast and no extended-precision evaluation.  Contraction is
// harmless because no expression below has the form x * y + z outside an
// explicit std::fma, so -ffp-contract=fast (the GPU compilers' default)
// cannot fuse anything.  std::fma must be a hardware instruction for the
// kernel to be fast; a libm fmaf is still exact.
//
// ffloat is a plain aggregate of two floats, deliberately without
// constructors or conversion constructors: with those, nvc++ compiled the
// OpenMP `teams loop` over elements across teams only, one thread per team
// with the per-element arrays and every operator temporary in shared memory
// (about 50x slower than float), while it mapped the float and double
// instantiations one element per thread.  A value is zeroed with `T{}` and
// built from a double with to_ffloat().

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

// True when the host compiler emits a fused multiply-add instruction for
// std::fma on float (GCC's FP_FAST_FMAF; clang only exposes the ISA macros).
// Diagnostic only: it says nothing about an offload device.
#if defined(FP_FAST_FMAF) || defined(__FMA__) || defined(__ARM_FEATURE_FMA)
constexpr bool ffloat_hardware_fma = true;
#else
constexpr bool ffloat_hardware_fma = false;
#endif

struct ffloat {
    float hi;
    float lo;

    explicit constexpr operator double() const { return static_cast<double>(hi) + static_cast<double>(lo); }
};

static_assert(sizeof(ffloat) == 2 * sizeof(float), "ffloat is two packed floats");
static_assert(std::is_aggregate_v<ffloat> && std::is_trivial_v<ffloat>,
              "ffloat stays a plain aggregate: the offload compilers privatise it like a scalar");

// Exact split of a double into the leading 24 bits and the next 24 bits
// (for |x| within float's range).
constexpr ffloat to_ffloat(double x)
{
    const float hi = static_cast<float>(x);
    return ffloat{hi, static_cast<float>(x - static_cast<double>(hi))};
}

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

// The hi parts are summed error-free, the lo parts in plain fp32, then the
// pair is renormalised.  11 flops.
inline ffloat operator+(ffloat a, ffloat b)
{
    ffloat s = ff::two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return ff::quick_two_sum(s.hi, s.lo);
}

inline ffloat operator-(ffloat a, ffloat b) { return a + (-b); }

// a.hi * b.hi exactly, plus the two cross terms; a.lo * b.lo is below the
// precision of the result and is dropped.  7 flops.
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
