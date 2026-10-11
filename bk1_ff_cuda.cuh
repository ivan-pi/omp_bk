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
// (grid-stride over the batches).  The body is a __device__ function used
// by the __global__ MassOperatorFF here and by the kernel functor of
// cuda/BK1_ff_cuda.cu, which takes the launch configuration as a template
// parameter (cuda::launch).  Within a batch a thread owns one point of
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
// Launch geometry: mass_operator_ff_traits<nq> gives the block size, a
// thread per point of the nq x nq plane of each element of the batch
// rounded up to whole warps (phases A, B, D and E use nm * nm or nq * nm
// points per element and leave the rest idle), the bytes of dynamic shared
// memory, the same count as the double kernel's (above 48 KB a plain
// launch needs cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicShared-
// MemorySize, bytes); cuda::launch takes cuda::non_portable), and the
// default batch.
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
// Phase E's second contraction (p into i) keeps both loops unrolled: the
// column ch[p] and the output yh[i] are both register arrays.
//
// Known cost left for later: the split is one fp64 subtract per loaded
// value, which on a 1/64-rate GPU is ~30% of the arithmetic at nq = 4 and
// ~13% at nq = 10 (a bit-level split with integer operations would avoid it).

#if defined(__CUDACC__)

#include <cstddef>

#include "float_float.h"

// The loops over a register column are always fully unrolled, which is
// what keeps the column in registers.  The loops over the contraction
// index are fully unrolled by default (FF_CUDA_CONTRACTION_UNROLL = 0),
// which lets ptxas hoist the basis and input loads of several iterations
// ahead of the arithmetic at the price of registers: 168 at nq = 8 and
// 10, 220 at nq = 9 on sm_90, which leaves 12 or 9 warps per SM.  A rolled
// contraction loop (factor 1) halves the throughput, and a register cap
// (-maxrregcount) spills inside the chains and halves it too.
// -DFF_CUDA_CONTRACTION_UNROLL=n unrolls the contraction loops by n
// instead, bounding the hoisting to n iterations.  With the full unroll,
// phase A loads the whole input column before its arithmetic (nm global
// loads in flight together, which the few warps per SM at high order
// need); with a partial factor it splits each value where it is used.
#ifndef FF_CUDA_CONTRACTION_UNROLL
#define FF_CUDA_CONTRACTION_UNROLL 0
#endif

namespace bk {

// The sizes of the kernel for one order and its launch geometry for a
// batch of elements per block: the sizes per element, the threads per
// element (one per point of the nq x nq plane), the block size, the
// dynamic shared memory (the split basis and two hi/lo box pairs) and the
// default batch, which fills about `threads` threads per block as
// ff_plane_batch of BK1_ff.cpp.
template <int nq_>
struct mass_operator_ff_traits {
    static_assert(nq_ >= 2, "nq = p + 2 with p >= 0");
    static constexpr int nq = nq_;               // quadrature points per direction
    static constexpr int nm = nq - 1;            // modes per direction
    static constexpr int nq3 = nq * nq * nq;     // quadrature points per element (JxW values)
    static constexpr int nm3 = nm * nm * nm;     // degrees of freedom per element (in, out values)
    static constexpr int basis_size = nm * nq;   // B(mode, quad point)
    static constexpr int plane = nq * nq;        // threads per element
    static constexpr int warp = 32;

    static constexpr int block_size(const int nelmtPerBatch)
    {
        return ((nelmtPerBatch * plane + warp - 1) / warp) * warp;
    }
    static constexpr std::size_t shared_floats(const int nelmtPerBatch)
    {
        return 2 * basis_size + 4 * static_cast<std::size_t>(nelmtPerBatch) * nq3;
    }
    static constexpr std::size_t shared_bytes(const int nelmtPerBatch)
    {
        return sizeof(float) * shared_floats(nelmtPerBatch);
    }
    static constexpr int default_batch(const int threads = 128)
    {
        return (threads / plane > 1) ? threads / plane : 1;
    }
};

// The kernel body, shared by the __global__ entry below and the kernel
// functor of cuda/BK1_ff_cuda.cu: `shared` is the block's dynamic shared
// memory, and the four counts are the block's rank and the grid size, the
// thread's rank in the block and the block size.
template <int nq>
__device__ inline void mass_operator_ff_body(
    const unsigned int nelmt, const unsigned int nelmtPerBatch,
    const double* __restrict__ d_basis,
    const double* __restrict__ d_JxW,
    const double* __restrict__ d_in,
    double* __restrict__ d_out,
    float* shared,
    const unsigned int block_rank, const unsigned int grid_size,
    const unsigned int thread_rank, const unsigned int block_size)
{
    using traits = mass_operator_ff_traits<nq>;
    constexpr int nm = traits::nm;
    constexpr int nq3 = traits::nq3;
    constexpr int nm3 = traits::nm3;
    // unroll factor of the contraction loops: full, or the compiler's choice
    static_assert(FF_CUDA_CONTRACTION_UNROLL >= 0, "unroll factor is 0 (full) or positive");
    [[maybe_unused]] constexpr int unroll_nm = (FF_CUDA_CONTRACTION_UNROLL == 0) ? nm : FF_CUDA_CONTRACTION_UNROLL;
    [[maybe_unused]] constexpr int unroll_nq = (FF_CUDA_CONTRACTION_UNROLL == 0) ? nq : FF_CUDA_CONTRACTION_UNROLL;

    // Shared memory: the split basis and the two hi/lo box pairs, laid out
    // as traits::shared_floats counts them.
    float* s_basis_hi = shared;
    float* s_basis_lo = s_basis_hi + nm * nq;
    float* s_wsp0_hi  = s_basis_lo + nm * nq;
    float* s_wsp0_lo  = s_wsp0_hi + nelmtPerBatch * nq3;
    float* s_wsp1_hi  = s_wsp0_lo + nelmtPerBatch * nq3;
    float* s_wsp1_lo  = s_wsp1_hi + nelmtPerBatch * nq3;

    // Accessors: the two boxes as hi and lo, nq^3 per element, the basis,
    // and (inside the batch loop) the batch's global arrays.  Each is a
    // lambda that captures its pointer by value and returns the element.
    // No __device__ on these lambdas (nor on decompose below): nvcc gives
    // the closure of a lambda the execution space of the innermost
    // enclosing function, so inside a __global__ function they are device
    // lambdas; an explicit annotation is only for lambdas written in host
    // code and passed to the device (--extended-lambda).
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
    for (int i = static_cast<int>(thread_rank); i < nm * nq; i += static_cast<int>(block_size)) {
        ff::split(d_basis[i], s_basis_hi[i], s_basis_lo[i]);
    }
    __syncthreads();

    const unsigned int nbatches = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch;
    for (unsigned int eb = block_rank; eb < nbatches; eb += grid_size) {

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
        for (int tid = static_cast<int>(thread_rank); tid < nb * nm * nm; tid += static_cast<int>(block_size)) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            if constexpr (FF_CUDA_CONTRACTION_UNROLL == 0) {
                // the whole input column is loaded before the arithmetic,
                // so that its nm global loads are in flight together
                float uh[nm];
                float ul[nm];
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    ff::split(in_at(e, i, j, k), uh[i], ul[i]);
                }
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    #pragma unroll
                    for (int p = 0; p < nq; ++p) {
                        ff::mul_add(ch[p], cl[p], uh[i], ul[i], basis_hi(i, p), basis_lo(i, p));
                    }
                }
            } else {
                // a partially unrolled contraction loop would push a column
                // array to local memory: split each value where it is used
                #pragma unroll unroll_nm
                for (int i = 0; i < nm; ++i) {
                    float uh;
                    float ul;
                    ff::split(in_at(e, i, j, k), uh, ul);
                    #pragma unroll
                    for (int p = 0; p < nq; ++p) {
                        ff::mul_add(ch[p], cl[p], uh, ul, basis_hi(i, p), basis_lo(i, p));
                    }
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
        for (int tid = static_cast<int>(thread_rank); tid < nb * nq * nm; tid += static_cast<int>(block_size)) {
            const auto [e, q, k] = decompose(tid, nq, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll unroll_nm
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
        for (int tid = static_cast<int>(thread_rank); tid < nb * nq * nq; tid += static_cast<int>(block_size)) {
            const auto [e, q, r] = decompose(tid, nq, nq);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll unroll_nm
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
        for (int tid = static_cast<int>(thread_rank); tid < nb * nq * nm; tid += static_cast<int>(block_size)) {
            const auto [e, q, k] = decompose(tid, nq, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll unroll_nq
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
        for (int tid = static_cast<int>(thread_rank); tid < nb * nm * nm; tid += static_cast<int>(block_size)) {
            const auto [e, j, k] = decompose(tid, nm, nm);
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                ch[p] = 0.0f;
                cl[p] = 0.0f;
            }
            #pragma unroll unroll_nq
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

// The kernel as a __global__ function: launch with
// mass_operator_ff_traits<nq>::shared_bytes(nelmtPerBatch) bytes of
// dynamic shared memory (cuda/check_ff_cuda.cpp runs this form on the host).
template <int nq>
__global__ void MassOperatorFF(
    const unsigned int nelmt, const unsigned int nelmtPerBatch,
    const double* __restrict__ d_basis,
    const double* __restrict__ d_JxW,
    const double* __restrict__ d_in,
    double* __restrict__ d_out)
{
    extern __shared__ float shared[];
    mass_operator_ff_body<nq>(nelmt, nelmtPerBatch, d_basis, d_JxW, d_in, d_out, shared,
                              blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

} // namespace bk

#endif // __CUDACC__

#endif // BK1_FF_CUDA_CUH
