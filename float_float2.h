#ifndef FLOAT_FLOAT2_H
#define FLOAT_FLOAT2_H

// float_float2.h -- the float-float helpers of float_float.h on float2
// pairs, for the experiment kernel MassOperatorFF2 of bk1_ff_cuda.cuh.
//
// The arithmetic is the same, operation for operation, as float_float.h:
// the same TwoSum, TwoProd, product, accumulation and renormalisation in
// the same order, so a kernel built on these helpers gives the same bits
// as one built on the separate hi/lo helpers.  What differs is the form:
// a pair is one float2 (x = hi, y = lo; NVIDIA's dbldbl.h keeps the head in
// y), the helpers take and return pairs by value in the style of that
// header, and on the device every operation is an explicit IEEE
// round-to-nearest intrinsic (__fadd_rn, __fmul_rn, __fmaf_rn), which no
// contraction or fast-math flag can rewrite.  Off the device (the host
// harness through cuda/cuda_shim.h) they are the plain operations and
// std::fma, as in float_float.h.
//
// The point of the pair form is the memory side of the kernel: a pair in
// shared memory is one 8-byte load or store instead of two 4-byte ones,
// and the register pair is one value for the compiler to carry.  Whether
// that is worth anything against the separate-array form is what the
// experiment measures.
//
// Device only in spirit: the header is meant for bk1_ff_cuda.cuh and the
// shim; the OpenMP kernels keep the separate hi/lo arrays of float_float.h.

#include <cmath>

namespace bk {
namespace ff2 {

__device__ __forceinline__ float add_rn(const float a, const float b)
{
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

__device__ __forceinline__ float sub_rn(const float a, const float b)
{
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

__device__ __forceinline__ float mul_rn(const float a, const float b)
{
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

__device__ __forceinline__ float fma_rn(const float a, const float b, const float c)
{
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// Exact split of a double into its leading 24 bits and the next 24 bits:
// hi = fl(x), lo = fl(x - hi), as float_float.h's split.
__device__ __forceinline__ float2 split(const double x)
{
#if defined(__CUDA_ARCH__)
    const float hi = __double2float_rn(x);
    const float lo = __double2float_rn(__dsub_rn(x, static_cast<double>(hi)));
#else
    const float hi = static_cast<float>(x);
    const float lo = static_cast<float>(x - static_cast<double>(hi));
#endif
    return make_float2(hi, lo);
}

// hi + lo as a double; exact.
__device__ __forceinline__ double join(const float2 a)
{
    return static_cast<double>(a.x) + static_cast<double>(a.y);
}

// (s, e) with s + e == a + b exactly, s = fl(a + b).  6 flops.
__device__ __forceinline__ float2 two_sum(const float a, const float b)
{
    const float s = add_rn(a, b);
    const float bb = sub_rn(s, a);
    const float e = add_rn(sub_rn(a, sub_rn(s, bb)), sub_rn(b, bb));
    return make_float2(s, e);
}

// Same result, 3 flops, valid only when |a| >= |b| or a == 0.
__device__ __forceinline__ float2 quick_two_sum(const float a, const float b)
{
    const float s = add_rn(a, b);
    const float e = sub_rn(b, sub_rn(s, a));
    return make_float2(s, e);
}

// (p, e) with p + e == a * b exactly, p = fl(a * b).  1 multiply + 1 fma.
__device__ __forceinline__ float2 two_prod(const float a, const float b)
{
    const float p = mul_rn(a, b);
    const float e = fma_rn(a, b, -p);
    return make_float2(p, e);
}

// a * b: a.x * b.x exactly, plus the two cross terms; a.y * b.y is dropped.
// 7 flops, as float_float.h's mul.
__device__ __forceinline__ float2 mul(const float2 a, const float2 b)
{
    const float2 p = two_prod(a.x, b.x);
    const float e = fma_rn(a.x, b.y, fma_rn(a.y, b.x, p.y));
    return quick_two_sum(p.x, e);
}

// t + a * b without renormalising the sum, 12 flops, as float_float.h's
// mul_acc: the product as in mul without its renormalisation, the hi parts
// summed error-free, the lo parts in plain fp32.  Renormalise once per
// column.
__device__ __forceinline__ float2 mul_acc(const float2 t, const float2 a, const float2 b)
{
    const float2 p = two_prod(a.x, b.x);
    const float pl = fma_rn(a.x, b.y, fma_rn(a.y, b.x, p.y));
    const float2 s = two_sum(t.x, p.x);
    return make_float2(s.x, add_rn(s.y, add_rn(t.y, pl)));
}

// The normalised pair of the same value, |y| <= ulp(x) / 2.  3 flops.
__device__ __forceinline__ float2 renormalize(const float2 t)
{
    return quick_two_sum(t.x, t.y);
}

} // namespace ff2
} // namespace bk

#endif // FLOAT_FLOAT2_H
