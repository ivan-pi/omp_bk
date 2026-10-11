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
// an element's plane and the column through it, held in registers, and each
// step contracts along that column with the basis read from shared memory
// as a warp broadcast.  The boxes between the steps are two pairs of
// hi/lo float arrays in dynamic shared memory with compact layouts:
//
//   step A (1+2): thread (e,j,k): in(i,j,k) from global -> contract i -> wsp1(e,p,j,k)
//   step B (3):   thread (e,p,k): wsp1(e,p,j,k)  -> contract j -> wsp0(e,q,p,k)
//   step C (4+5): thread (e,q,p): wsp0(e,q,p,k)  -> contract k, * JxW -> wsp1(e,r,q,p)
//   step D (6):   thread (e,q,p): wsp1(e,r,q,p)  -> contract r -> wsp0(e,k,q,p)
//   step E (7):   thread (e,k,p): wsp0(e,k,q,p)  -> contract q -> wsp1(e,k,j,p)
//   step F (8+9): thread (e,j,k): wsp1(e,k,j,p)  -> contract p -> out(i,j,k) to global
//
// Steps A and F read and write global memory directly, so the copy steps of
// the double kernel and two of its barriers are gone.  Every output is still
// summed in index order, with the same operand order as the OpenMP kernels
// of BK1_ff.cpp.  The boxes and the batch's global arrays are addressed
// through accessor lambdas that capture the pointers by value, one pair per
// layout for hi and lo, so no index arithmetic appears in the steps.
//
// Shared memory: mass_operator_ff_shared_bytes(nq, nelmtPerBatch) bytes of
// dynamic shared memory, the same byte count as the double kernel's.  Above
// 48 KB the launch needs cudaFuncSetAttribute(...,
// cudaFuncAttributeMaxDynamicSharedMemorySize, bytes).  A block of
// nelmtPerBatch * nq * nq threads (rounded up to whole warps) gives every
// point of step C a thread; the other steps use nm * nm or nm * nq points
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
// Known costs left for later: the split is one fp64 subtract per loaded
// value, which on a 1/64-rate GPU is ~30% of the arithmetic at nq = 4 and
// ~13% at nq = 10 (a bit-level split with integer operations would avoid
// it); the column loads of steps C and F stride by nm and nq across the
// warp and cause bank conflicts for even nm or nq.

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

    // Accessors: the boxes with the layout of each step, as hi and lo, the
    // basis, and (inside the batch loop) the batch's global arrays.  Each is
    // a lambda that captures its pointer by value and returns the element.
    const auto w1_pjk_hi = [=](int e, int p, int j, int k) -> float& { return s_wsp1_hi[((e * nq + p) * nm + j) * nm + k]; };
    const auto w1_pjk_lo = [=](int e, int p, int j, int k) -> float& { return s_wsp1_lo[((e * nq + p) * nm + j) * nm + k]; };
    const auto w0_qpk_hi = [=](int e, int q, int p, int k) -> float& { return s_wsp0_hi[((e * nq + q) * nq + p) * nm + k]; };
    const auto w0_qpk_lo = [=](int e, int q, int p, int k) -> float& { return s_wsp0_lo[((e * nq + q) * nq + p) * nm + k]; };
    const auto w1_rqp_hi = [=](int e, int r, int q, int p) -> float& { return s_wsp1_hi[((e * nq + r) * nq + q) * nq + p]; };
    const auto w1_rqp_lo = [=](int e, int r, int q, int p) -> float& { return s_wsp1_lo[((e * nq + r) * nq + q) * nq + p]; };
    const auto w0_kqp_hi = [=](int e, int k, int q, int p) -> float& { return s_wsp0_hi[((e * nm + k) * nq + q) * nq + p]; };
    const auto w0_kqp_lo = [=](int e, int k, int q, int p) -> float& { return s_wsp0_lo[((e * nm + k) * nq + q) * nq + p]; };
    const auto w1_kjp_hi = [=](int e, int k, int j, int p) -> float& { return s_wsp1_hi[((e * nm + k) * nm + j) * nq + p]; };
    const auto w1_kjp_lo = [=](int e, int k, int j, int p) -> float& { return s_wsp1_lo[((e * nm + k) * nm + j) * nq + p]; };
    const auto basis_hi  = [=](int i, int p) -> float { return s_basis_hi[i * nq + p]; };
    const auto basis_lo  = [=](int i, int p) -> float { return s_basis_lo[i * nq + p]; };

    // A thread's point: element e of the batch and (a, b) in a plane of
    // na x nb points per element, from the thread index of a step.
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

    // The register column: the input of a step (nm or nq values) as hi and lo.
    float rh[nq];
    float rl[nq];

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
        const auto JxW_at = [=](int e, int r, int q, int p) -> double  { return b_JxW[((e * nq + r) * nq + q) * nq + p]; };
        const auto out_at = [=](int e, int i, int j, int k) -> double& { return b_out[((e * nm + i) * nm + j) * nm + k]; };

        // step A : load and split the column in(:, j, k), contract direction 0
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                ff::split(in_at(e, i, j, k), rh[i], rl[i]);
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    ff::mul_add(th, tl, rh[i], rl[i], basis_hi(i, p), basis_lo(i, p));
                }
                w1_pjk_hi(e, p, j, k) = th;
                w1_pjk_lo(e, p, j, k) = tl;
            }
        }
        __syncthreads();

        // step B : direction 1
        for (int tid = threadIdx.x; tid < nb * nq * nm; tid += blockDim.x) {
            const auto [e, p, k] = decompose(tid, nq, nm);
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                rh[j] = w1_pjk_hi(e, p, j, k);
                rl[j] = w1_pjk_lo(e, p, j, k);
            }
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int j = 0; j < nm; ++j) {
                    ff::mul_add(th, tl, rh[j], rl[j], basis_hi(j, q), basis_lo(j, q));
                }
                w0_qpk_hi(e, q, p, k) = th;
                w0_qpk_lo(e, q, p, k) = tl;
            }
        }
        __syncthreads();

        // step C : direction 2, then the weights and determinant of Jacobi,
        // each JxW value split at its single use
        for (int tid = threadIdx.x; tid < nb * nq * nq; tid += blockDim.x) {
            const auto [e, q, p] = decompose(tid, nq, nq);
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                rh[k] = w0_qpk_hi(e, q, p, k);
                rl[k] = w0_qpk_lo(e, q, p, k);
            }
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int k = 0; k < nm; ++k) {
                    ff::mul_add(th, tl, rh[k], rl[k], basis_hi(k, r), basis_lo(k, r));
                }
                float jh;
                float jl;
                ff::split(JxW_at(e, r, q, p), jh, jl);
                ff::mul(th, tl, jh, jl);
                w1_rqp_hi(e, r, q, p) = th;
                w1_rqp_lo(e, r, q, p) = tl;
            }
        }
        __syncthreads();

        // Reverse operations

        // step D : direction 2 back
        for (int tid = threadIdx.x; tid < nb * nq * nq; tid += blockDim.x) {
            const auto [e, q, p] = decompose(tid, nq, nq);
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                rh[r] = w1_rqp_hi(e, r, q, p);
                rl[r] = w1_rqp_lo(e, r, q, p);
            }
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int r = 0; r < nq; ++r) {
                    ff::mul_add(th, tl, rh[r], rl[r], basis_hi(k, r), basis_lo(k, r));
                }
                w0_kqp_hi(e, k, q, p) = th;
                w0_kqp_lo(e, k, q, p) = tl;
            }
        }
        __syncthreads();

        // step E : direction 1 back
        for (int tid = threadIdx.x; tid < nb * nm * nq; tid += blockDim.x) {
            const auto [e, k, p] = decompose(tid, nm, nq);
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                rh[q] = w0_kqp_hi(e, k, q, p);
                rl[q] = w0_kqp_lo(e, k, q, p);
            }
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    ff::mul_add(th, tl, rh[q], rl[q], basis_hi(j, q), basis_lo(j, q));
                }
                w1_kjp_hi(e, k, j, p) = th;
                w1_kjp_lo(e, k, j, p) = tl;
            }
        }
        __syncthreads();

        // step F : direction 0 back, join and store the column out(:, j, k)
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                rh[p] = w1_kjp_hi(e, k, j, p);
                rl[p] = w1_kjp_lo(e, k, j, p);
            }
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(th, tl, rh[p], rl[p], basis_hi(i, p), basis_lo(i, p));
                }
                out_at(e, i, j, k) = ff::join(th, tl);
            }
        }
        // the next batch's step A overwrites wsp1 while step F may still read it
        __syncthreads();
    }
}

} // namespace bk

#endif // __CUDACC__

#endif // BK1_FF_CUDA_CUH
