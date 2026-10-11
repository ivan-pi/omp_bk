#ifndef FLOAT_FLOAT_H
#define FLOAT_FLOAT_H

// float_float.h -- "float-float" (double-float) arithmetic helpers.
//
// Private implementation header of BK1_ff.cpp: nothing else includes it and
// no type of it appears in an interface.  The kernel takes and returns IEEE
// doubles and only internally carries each value as a float pair.
//
// A float-float value is the unevaluated sum hi + lo of two IEEE binary32
// numbers with |lo| <= ulp(hi) / 2, which carries about 48 significant bits
// (2 x 24, a few bits short of double's 53).  The exponent range is that of
// float, so values must stay within ~1e-38 .. 3e38 and no denormal arithmetic
// is guaranteed.  Every operation is a chain of fp32 adds, multiplies and
// fused multiply-adds: on hardware whose fp64 throughput is a fraction of
// fp32 (consumer GPUs) this is the usual way to get near-double accuracy at
// near-fp32 cost.
//
// There is no pair type: hi and lo are separate float scalars (or separate
// arrays in the kernel) and every helper takes them as separate arguments and
// writes its result through references.  After inlining the compiler sees
// plain float SSA values and plain 4-byte array accesses, with no struct to
// keep in local memory.
//
// Costs in fp32 flops: mul is 7 and mul_add 15 (a multiply of 7 followed by
// an add of 11 would be 18; mul_add skips the renormalisation of the product,
// whose lo part is then at most a few ulp of its hi and is absorbed by the
// lo sum of the addition).  The addition is the QD library's sloppy_add: the
// rounding error of lo + lo is dropped, which costs digits only when the sum
// cancels to far below its larger term.  On BK1 this reaches ~3e-15 relative
// error against double.
//
// The conversions cost fp64 work: split is one fp64 subtract (plus the
// rounding to float) and join one fp64 add.  On a GPU with fp64 at 1/64 of
// the fp32 rate this is of the order of 10% of the kernel's fp32 work; a
// bit-level split with integer operations would avoid it.
//
// The building blocks are the error-free transformations TwoSum (Knuth) and
// TwoProd (Dekker, via fma), which only hold under strict IEEE rounding: no
// -ffast-math / -Ofast and no extended-precision evaluation.  Contraction is
// harmless because no expression below has the form x * y + z outside an
// explicit std::fma, so -ffp-contract=fast (the GPU compilers' default)
// cannot fuse anything.  std::fma must be a hardware instruction for the
// kernel to be fast; a libm fmaf is still exact.

#include <cfloat>
#include <cmath>
#include <limits>

#if defined(__FAST_MATH__)
#error "float-float arithmetic needs IEEE semantics: do not compile with -ffast-math / -Ofast"
#endif

namespace bk {
namespace ff {

static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE binary32");
static_assert(std::numeric_limits<double>::is_iec559, "double must be IEEE binary64");
static_assert(FLT_EVAL_METHOD == 0, "float expressions must be evaluated in float (no x87 excess precision)");

// True when the host compiler emits a fused multiply-add instruction for
// std::fma on float (GCC's FP_FAST_FMAF; clang only exposes the ISA macros).
// Diagnostic only: it says nothing about an offload device.
#if defined(FP_FAST_FMAF) || defined(__FMA__) || defined(__ARM_FEATURE_FMA)
constexpr bool hardware_fma = true;
#else
constexpr bool hardware_fma = false;
#endif

// Exact split of a double into its leading 24 bits and the next 24 bits
// (for |x| within float's range): hi = fl(x), lo = fl(x - hi) where x - hi
// is exact in double.
inline void split(const double x, float& hi, float& lo)
{
    hi = static_cast<float>(x);
    lo = static_cast<float>(x - static_cast<double>(hi));
}

// hi + lo as a double; exact, both parts fit in 53 bits.
inline double join(const float hi, const float lo)
{
    return static_cast<double>(hi) + static_cast<double>(lo);
}

// (s, e) with s + e == a + b exactly, s = fl(a + b).  6 flops.
inline void two_sum(const float a, const float b, float& s, float& e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Same result, 3 flops, valid only when |a| >= |b| or a == 0.
inline void quick_two_sum(const float a, const float b, float& s, float& e)
{
    s = a + b;
    e = b - (s - a);
}

// (p, e) with p + e == a * b exactly, p = fl(a * b).  1 multiply + 1 fma.
inline void two_prod(const float a, const float b, float& p, float& e)
{
    p = a * b;
    e = std::fma(a, b, -p);
}

// (ah, al) *= (bh, bl): ah * bh exactly, plus the two cross terms; al * bl is
// below the precision of the result and is dropped.  7 flops.
inline void mul(float& ah, float& al, const float bh, const float bl)
{
    float p;
    float e;
    two_prod(ah, bh, p, e);
    e = std::fma(ah, bl, std::fma(al, bh, e));
    quick_two_sum(p, e, ah, al);
}

// (th, tl) += (ah, al) * (bh, bl): the product as in mul but without its
// renormalisation, then the hi parts are summed error-free, the lo parts in
// plain fp32, and the pair is renormalised once.  15 flops.
inline void mul_add(float& th, float& tl,
                    const float ah, const float al,
                    const float bh, const float bl)
{
    float ph;
    float pl;
    two_prod(ah, bh, ph, pl);
    pl = std::fma(ah, bl, std::fma(al, bh, pl));
    float sh;
    float sl;
    two_sum(th, ph, sh, sl);
    sl += tl + pl;
    quick_two_sum(sh, sl, th, tl);
}

} // namespace ff
} // namespace bk

#endif // FLOAT_FLOAT_H
