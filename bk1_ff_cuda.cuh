#ifndef BK1_FF_CUDA_CUH
#define BK1_FF_CUDA_CUH

// bk1_ff_cuda.cuh -- the BK1 mass operator in float-float arithmetic as a
// CUDA kernel, with double storage at the boundary.
//
// MassOperatorFF<nq> has the interface of a double-precision kernel: the
// basis, the weights, the input and the output are IEEE doubles in global
// memory.  Inside, every value is split into a float pair (hi + lo, ~48
// significant bits) as it is loaded, the sum factorization runs on float
// pairs with the helpers of float_float.h, and the result is joined back to
// a double as it is stored.  The point is hardware whose fp64 rate is a
// small fraction of fp32: a float-float multiply-add is 15 fp32
// instructions, cheaper than fp64 at 1/32 or 1/64 of the fp32 rate, and the
// kernel moves the same 8 bytes per value as double.
//
// Mapping: one block works through batches of nelmtPerBatch elements
// (grid-stride over the batches).  Within a batch a thread owns one point of
// an element's plane and the whole column through it along the first index,
// held in registers as hi and lo of nq values, and the five phases are those
// of the OpenMP plane kernel (plane_phase_a..e in BK1_ff.cpp): the
// contraction index is the outer loop, the column index the inner one, and
// the basis is read from shared memory as a warp broadcast.  The boxes
// between the phases are two pairs of hi/lo float arrays in dynamic shared
// memory, each nq^3 per element with the one layout (a, b, c) ->
// (a * nq + b) * nq + c of the OpenMP kernels:
//
//   A: thread (e,j,k): in(i,j,k) from global -> contract i -> w0(p,j,k)
//   B: thread (e,q,k): w0(p,j,k)  -> contract j -> w1(p,q,k)
//   C: thread (e,q,r): w1(p,q,k)  -> contract k, * JxW(p,q,r) -> w0(p,q,r)
//   D: thread (e,q,k): w0(p,q,r)  -> contract r -> w1(p,q,k)
//   E: thread (e,j,k): w1(p,q,k)  -> contract q -> x[p], contract p -> out(i,j,k)
//
// Phases A and E read and write global memory directly, so the copy steps of
// the double kernel and two of its barriers are gone.  Every output is
// summed in index order with the operand order of the OpenMP kernels, so the
// results agree with them bit for bit.  The boxes and the batch's global
// arrays are addressed through accessor lambdas that capture the pointers by
// value and return the element.
//
// Shared memory: mass_operator_ff_shared_bytes(nq, nelmtPerBatch) bytes of
// dynamic shared memory, the same byte count as the double kernel's.  Above
// 48 KB the launch needs cudaFuncSetAttribute(...,
// cudaFuncAttributeMaxDynamicSharedMemorySize, bytes).  A block of
// nelmtPerBatch * nq * nq threads (rounded up to whole warps) gives every
// point of phase C a thread; the other phases use nm * nm or nq * nm points
// per element and leave the rest idle.
//
// Flags: the error-free transformations need no option.  nvcc's default
// -fmad=true cannot contract anything because no expression has the form
// a * b + c outside an explicit fmaf; --use_fast_math is harmless for adds
// and multiplies, but its -ftz flushes subnormal lo parts, which only
// matters for values with |hi| below ~1e-31.
//
// Example build:  nvcc -O3 -std=c++17 -arch=sm_90 -x cu -c driver.cpp
// (the driver includes this header and float_float.h).
//
// Known cost left for later: the split is one fp64 subtract per loaded
// value, which on a 1/64-rate GPU is ~30% of the arithmetic at nq = 4 and
// ~13% at nq = 10 (a bit-level split with integer operations would avoid it).

#if defined(__CUDACC__)

#include <cstddef>

#include "float_float.h"

namespace bk {

// Bytes of dynamic shared memory for MassOperatorFF<nq> with a batch of
// nelmtPerBatch elements: the split basis and two hi/lo box pairs.
constexpr std::size_t mass_operator_ff_shared_bytes(const int nq, const int nelmtPerBatch)
{
    return sizeof(float) * (2 * (nq - 1) * nq + 4 * static_cast<std::size_t>(nelmtPerBatch) * nq * nq * nq);
}

template <int nq>
__global__ void MassOperatorFF(
    const unsigned int nelmt, const unsigned int nelmtPerBatch,
    const double* __restrict__ d_basis,
    const double* __restrict__ d_JxW,
    const double* __restrict__ d_in,
    double* __restrict__ d_out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;

    // Shared memory: the split basis and the two hi/lo box pairs.
    extern __shared__ float shared[];
    float* s_basis_hi = shared;
    float* s_basis_lo = s_basis_hi + nm * nq;
    float* s_wsp0_hi  = s_basis_lo + nm * nq;
    float* s_wsp0_lo  = s_wsp0_hi + nelmtPerBatch * nq3;
    float* s_wsp1_hi  = s_wsp0_lo + nelmtPerBatch * nq3;
    float* s_wsp1_lo  = s_wsp1_hi + nelmtPerBatch * nq3;

    // Accessors: the two boxes as hi and lo, nq^3 per element, the basis,
    // and (inside the batch loop) the batch's global arrays.  Each is a
    // lambda that captures its pointer by value and returns the element.
    const auto w0_hi = [=](int e, int a, int b, int c) -> float& { return s_wsp0_hi[((e * nq + a) * nq + b) * nq + c]; };
    const auto w0_lo = [=](int e, int a, int b, int c) -> float& { return s_wsp0_lo[((e * nq + a) * nq + b) * nq + c]; };
    const auto w1_hi = [=](int e, int a, int b, int c) -> float& { return s_wsp1_hi[((e * nq + a) * nq + b) * nq + c]; };
    const auto w1_lo = [=](int e, int a, int b, int c) -> float& { return s_wsp1_lo[((e * nq + a) * nq + b) * nq + c]; };
    const auto basis_hi = [=](int i, int p) -> float { return s_basis_hi[i * nq + p]; };
    const auto basis_lo = [=](int i, int p) -> float { return s_basis_lo[i * nq + p]; };

    // A thread's point: element e of the batch and (a, b) in a plane of
    // na x nb points per element, from the thread index of a phase.
    struct Point {
        int e;
        int a;
        int b;
    };
    const auto decompose = [](int tid, int na, int nb) -> Point {
        const int e = tid / (na * nb);
        const int r = tid - e * (na * nb);
        return Point{e, r / nb, r - (r / nb) * nb};
    };

    // The register column: the output of a phase as hi and lo, and the
    // input column of phase A.
    float ch[nq];
    float cl[nq];

    // The basis, split once per block.
    for (int i = threadIdx.x; i < nm * nq; i += blockDim.x) {
        ff::split(d_basis[i], s_basis_hi[i], s_basis_lo[i]);
    }
    __syncthreads();

    const unsigned int nbatches = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch;
    for (unsigned int eb = blockIdx.x; eb < nbatches; eb += gridDim.x) {

        // the last batch may be short
        const int remaining = static_cast<int>(nelmt - eb * nelmtPerBatch);
        const int nb = (static_cast<int>(nelmtPerBatch) < remaining) ? static_cast<int>(nelmtPerBatch) : remaining;

        const double* __restrict__ b_in  = d_in  + static_cast<std::size_t>(eb) * nelmtPerBatch * nm3;
        const double* __restrict__ b_JxW = d_JxW + static_cast<std::size_t>(eb) * nelmtPerBatch * nq3;
        double*       __restrict__ b_out = d_out + static_cast<std::size_t>(eb) * nelmtPerBatch * nm3;
        const auto in_at  = [=](int e, int i, int j, int k) -> double  { return b_in [((e * nm + i) * nm + j) * nm + k]; };
        const auto JxW_at = [=](int e, int p, int q, int r) -> double  { return b_JxW[((e * nq + p) * nq + q) * nq + r]; };
        const auto out_at = [=](int e, int i, int j, int k) -> double& { return b_out[((e * nm + i) * nm + j) * nm + k]; };

        // phase A : load and split the column in(:, j, k), contract direction 0
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            float uh[nm];
            float ul[nm];
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                ff::split(in_at(e, i, j, k), uh[i], ul[i]);
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(ch[p], cl[p], uh[i], ul[i], basis_hi(i, p), basis_lo(i, p));
                }
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                w0_hi(e, p, j, k) = ch[p];
                w0_lo(e, p, j, k) = cl[p];
            }
        }
        __syncthreads();

        // phase B : contract direction 1
        for (int tid = threadIdx.x; tid < nb * nq * nm; tid += blockDim.x) {
            const auto [e, q, k] = decompose(tid, nq, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                const float bh = basis_hi(j, q);
                const float bl = basis_lo(j, q);
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(ch[p], cl[p], w0_hi(e, p, j, k), w0_lo(e, p, j, k), bh, bl);
                }
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                w1_hi(e, p, q, k) = ch[p];
                w1_lo(e, p, q, k) = cl[p];
            }
        }
        __syncthreads();

        // phase C : contract direction 2, then the weights and determinant of
        // Jacobi, each JxW value split at its single use
        for (int tid = threadIdx.x; tid < nb * nq * nq; tid += blockDim.x) {
            const auto [e, q, r] = decompose(tid, nq, nq);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                const float bh = basis_hi(k, r);
                const float bl = basis_lo(k, r);
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(ch[p], cl[p], w1_hi(e, p, q, k), w1_lo(e, p, q, k), bh, bl);
                }
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                float jh;
                float jl;
                ff::split(JxW_at(e, p, q, r), jh, jl);
                ff::mul(ch[p], cl[p], jh, jl);
                w0_hi(e, p, q, r) = ch[p];
                w0_lo(e, p, q, r) = cl[p];
            }
        }
        __syncthreads();

        // phase D : contract direction 2 back
        for (int tid = threadIdx.x; tid < nb * nq * nm; tid += blockDim.x) {
            const auto [e, q, k] = decompose(tid, nq, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                const float bh = basis_hi(k, r);
                const float bl = basis_lo(k, r);
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(ch[p], cl[p], w0_hi(e, p, q, r), w0_lo(e, p, q, r), bh, bl);
                }
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                w1_hi(e, p, q, k) = ch[p];
                w1_lo(e, p, q, k) = cl[p];
            }
        }
        __syncthreads();

        // phase E : contract direction 1 back, then direction 0 back in
        // registers, join and store the column out(:, j, k)
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                const float bh = basis_hi(j, q);
                const float bl = basis_lo(j, q);
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(ch[p], cl[p], w1_hi(e, p, q, k), w1_lo(e, p, q, k), bh, bl);
                }
            }
            float yh[nm];
            float yl[nm];
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                yh[i] = 0.0f;
                yl[i] = 0.0f;
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    ff::mul_add(yh[i], yl[i], ch[p], cl[p], basis_hi(i, p), basis_lo(i, p));
                }
            }
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                out_at(e, i, j, k) = ff::join(yh[i], yl[i]);
            }
        }
        // the next batch's phase A overwrites w0 while phase E may still read w1,
        // and phase B of the next batch reads w0: one barrier covers both
        __syncthreads();
    }
}

} // namespace bk

#endif // __CUDACC__

#endif // BK1_FF_CUDA_CUH
