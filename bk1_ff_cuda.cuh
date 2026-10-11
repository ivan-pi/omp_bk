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
// summed in index order.  Compared with the OpenMP kernels of BK1_ff.cpp the
// operand order of the cross terms is the same, so the results agree bit for
// bit when the batch layout agrees; this is not required.
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

    // Indexing, in one place.  idx(e, a, b, c, na, nb, nc) is the offset of
    // point (a, b, c) of element e in a box of na x nb x nc per element;
    // the named helpers are the layouts of the steps above.
    const auto idx = [](int e, int a, int b, int c, int na, int nb, int nc) {
        return ((e * na + a) * nb + b) * nc + c;
    };
    const auto basis_at = [](int i, int p) { return i * nq + p; };
    const auto in_ijk   = [&](int e, int i, int j, int k) { return idx(e, i, j, k, nm, nm, nm); };
    const auto wsp_pjk  = [&](int e, int p, int j, int k) { return idx(e, p, j, k, nq, nm, nm); };
    const auto wsp_qpk  = [&](int e, int q, int p, int k) { return idx(e, q, p, k, nq, nq, nm); };
    const auto wsp_rqp  = [&](int e, int r, int q, int p) { return idx(e, r, q, p, nq, nq, nq); };
    const auto wsp_kqp  = [&](int e, int k, int q, int p) { return idx(e, k, q, p, nm, nq, nq); };
    const auto wsp_kjp  = [&](int e, int k, int j, int p) { return idx(e, k, j, p, nm, nm, nq); };
    // tid -> (e, a, b) over a plane of na x nb points per element
    const auto decompose = [](int tid, int na, int nb, int& e, int& a, int& b) {
        e = tid / (na * nb);
        const int r = tid - e * na * nb;
        a = r / nb;
        b = r - a * nb;
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

        // step A : load and split the column in(:, j, k), contract direction 0
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            int e;
            int j;
            int k;
            decompose(tid, nm, nm, e, j, k);
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                ff::split(b_in[in_ijk(e, i, j, k)], rh[i], rl[i]);
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    ff::mul_add(th, tl, rh[i], rl[i], s_basis_hi[basis_at(i, p)], s_basis_lo[basis_at(i, p)]);
                }
                s_wsp1_hi[wsp_pjk(e, p, j, k)] = th;
                s_wsp1_lo[wsp_pjk(e, p, j, k)] = tl;
            }
        }
        __syncthreads();

        // step B : direction 1
        for (int tid = threadIdx.x; tid < nb * nq * nm; tid += blockDim.x) {
            int e;
            int p;
            int k;
            decompose(tid, nq, nm, e, p, k);
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                rh[j] = s_wsp1_hi[wsp_pjk(e, p, j, k)];
                rl[j] = s_wsp1_lo[wsp_pjk(e, p, j, k)];
            }
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int j = 0; j < nm; ++j) {
                    ff::mul_add(th, tl, rh[j], rl[j], s_basis_hi[basis_at(j, q)], s_basis_lo[basis_at(j, q)]);
                }
                s_wsp0_hi[wsp_qpk(e, q, p, k)] = th;
                s_wsp0_lo[wsp_qpk(e, q, p, k)] = tl;
            }
        }
        __syncthreads();

        // step C : direction 2, then the weights and determinant of Jacobi,
        // each JxW value split at its single use
        for (int tid = threadIdx.x; tid < nb * nq * nq; tid += blockDim.x) {
            int e;
            int q;
            int p;
            decompose(tid, nq, nq, e, q, p);
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                rh[k] = s_wsp0_hi[wsp_qpk(e, q, p, k)];
                rl[k] = s_wsp0_lo[wsp_qpk(e, q, p, k)];
            }
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int k = 0; k < nm; ++k) {
                    ff::mul_add(th, tl, rh[k], rl[k], s_basis_hi[basis_at(k, r)], s_basis_lo[basis_at(k, r)]);
                }
                float jh;
                float jl;
                ff::split(b_JxW[wsp_rqp(e, r, q, p)], jh, jl);
                ff::mul(th, tl, jh, jl);
                s_wsp1_hi[wsp_rqp(e, r, q, p)] = th;
                s_wsp1_lo[wsp_rqp(e, r, q, p)] = tl;
            }
        }
        __syncthreads();

        // Reverse operations

        // step D : direction 2 back
        for (int tid = threadIdx.x; tid < nb * nq * nq; tid += blockDim.x) {
            int e;
            int q;
            int p;
            decompose(tid, nq, nq, e, q, p);
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                rh[r] = s_wsp1_hi[wsp_rqp(e, r, q, p)];
                rl[r] = s_wsp1_lo[wsp_rqp(e, r, q, p)];
            }
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int r = 0; r < nq; ++r) {
                    ff::mul_add(th, tl, rh[r], rl[r], s_basis_hi[basis_at(k, r)], s_basis_lo[basis_at(k, r)]);
                }
                s_wsp0_hi[wsp_kqp(e, k, q, p)] = th;
                s_wsp0_lo[wsp_kqp(e, k, q, p)] = tl;
            }
        }
        __syncthreads();

        // step E : direction 1 back
        for (int tid = threadIdx.x; tid < nb * nm * nq; tid += blockDim.x) {
            int e;
            int k;
            int p;
            decompose(tid, nm, nq, e, k, p);
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                rh[q] = s_wsp0_hi[wsp_kqp(e, k, q, p)];
                rl[q] = s_wsp0_lo[wsp_kqp(e, k, q, p)];
            }
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    ff::mul_add(th, tl, rh[q], rl[q], s_basis_hi[basis_at(j, q)], s_basis_lo[basis_at(j, q)]);
                }
                s_wsp1_hi[wsp_kjp(e, k, j, p)] = th;
                s_wsp1_lo[wsp_kjp(e, k, j, p)] = tl;
            }
        }
        __syncthreads();

        // step F : direction 0 back, join and store the column out(:, j, k)
        for (int tid = threadIdx.x; tid < nb * nm * nm; tid += blockDim.x) {
            int e;
            int j;
            int k;
            decompose(tid, nm, nm, e, j, k);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                rh[p] = s_wsp1_hi[wsp_kjp(e, k, j, p)];
                rl[p] = s_wsp1_lo[wsp_kjp(e, k, j, p)];
            }
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                float th = 0.0f;
                float tl = 0.0f;
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    ff::mul_add(th, tl, rh[p], rl[p], s_basis_hi[basis_at(i, p)], s_basis_lo[basis_at(i, p)]);
                }
                b_out[in_ijk(e, i, j, k)] = ff::join(th, tl);
            }
        }
        // the next batch's step A overwrites wsp1 while step F may still read it
        __syncthreads();
    }
}

} // namespace bk

#endif // __CUDACC__

#endif // BK1_FF_CUDA_CUH
