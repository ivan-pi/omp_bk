// bk1_gpu_steps.h -- the seven contraction steps of the CUDA/Kokkos BK1 kernel
// structure, one work item per call (tid = the flattened thread index of
// the step), shared by the OpenMP (grid_stride.h) and SYCL
// (sycl/bk1_sycl.cpp) kernels.  No OpenMP or SYCL dependency: plain
// functions on pointers, inlined into the kernels.  Every pointer is
// __restrict__: the input, the output box and the basis never alias.
//
// Box layouts per step (e = element of the batch):
//   in   (e, i, j, k)   s2 ->  w1 (e, p, j, k)   s3 -> w0 (e, q, p, k)
//   s4   -> w1 (e, r, q, p) * JxW                s6 -> w0 (e, k, q, p)
//   s7   -> w1 (e, k, j, p)                      s8 -> w0 (e, i, j, k) = out
// Every output is summed in the index order of bk::SumFactorization.

#ifndef BK1_GPU_STEPS_H
#define BK1_GPU_STEPS_H

#include "bk1_gpu.h"

namespace bk1gpu {

template <typename T, int nq>
struct gs_steps {
    static constexpr int nm = nq - 1;

    // step 2 : direction 0, thread (e, j, k): w0(e,i,j,k) -> w1(e,p,j,k)
    static void s2(const int tid, const T* __restrict__ B, const T* __restrict__ w0, T* __restrict__ w1)
    {
        const int e = tid / (nm * nm);
        const int j = tid % (nm * nm) / nm;
        const int k = tid % nm;
        T reg[nm];
        UNROLL
        for (int i = 0; i < nm; ++i) {
            reg[i] = w0[e * nm * nm * nm + i * nm * nm + j * nm + k];
        }
        UNROLL
        for (int p = 0; p < nq; ++p) {
            T tmp = 0;
            UNROLL
            for (int i = 0; i < nm; ++i) {
                tmp += reg[i] * B[i * nq + p];
            }
            w1[e * nq * nm * nm + p * nm * nm + j * nm + k] = tmp;
        }
    }
    // step 3 : direction 1, thread (e, p, k): w1(e,p,j,k) -> w0(e,q,p,k)
    static void s3(const int tid, const T* __restrict__ B, const T* __restrict__ w1, T* __restrict__ w0)
    {
        const int e = tid / (nq * nm);
        const int p = tid % (nq * nm) / nm;
        const int k = tid % nm;
        T reg[nm];
        UNROLL
        for (int j = 0; j < nm; ++j) {
            reg[j] = w1[e * nq * nm * nm + p * nm * nm + j * nm + k];
        }
        UNROLL
        for (int q = 0; q < nq; ++q) {
            T tmp = 0;
            UNROLL
            for (int j = 0; j < nm; ++j) {
                tmp += reg[j] * B[j * nq + q];
            }
            w0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
        }
    }
    // steps 4+5 : direction 2 and JxW, thread (e, q, p): w0(e,q,p,k) -> w1(e,r,q,p)
    static void s4(const int tid, const T* __restrict__ B, const T* __restrict__ JxW, const T* __restrict__ w0, T* __restrict__ w1)
    {
        const int e = tid / (nq * nq);
        const int q = tid % (nq * nq) / nq;
        const int p = tid % nq;
        T reg[nm];
        UNROLL
        for (int k = 0; k < nm; ++k) {
            reg[k] = w0[e * nq * nq * nm + q * nq * nm + p * nm + k];
        }
        UNROLL
        for (int r = 0; r < nq; ++r) {
            T tmp = 0;
            UNROLL
            for (int k = 0; k < nm; ++k) {
                tmp += reg[k] * B[k * nq + r];
            }
            // JxW(e, p, q, r) of the element, as in SumFactorization
            w1[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                tmp * JxW[e * nq * nq * nq + (p * nq + q) * nq + r];
        }
    }
    // step 4 without JxW (BK3's interpolation to the points), thread (e, q, p):
    // w0(e,q,p,k) -> w1(e,r,q,p)
    static void s4u(const int tid, const T* __restrict__ B, const T* __restrict__ w0, T* __restrict__ w1)
    {
        const int e = tid / (nq * nq);
        const int q = tid % (nq * nq) / nq;
        const int p = tid % nq;
        T reg[nm];
        UNROLL
        for (int k = 0; k < nm; ++k) {
            reg[k] = w0[e * nq * nq * nm + q * nq * nm + p * nm + k];
        }
        UNROLL
        for (int r = 0; r < nq; ++r) {
            T tmp = 0;
            UNROLL
            for (int k = 0; k < nm; ++k) {
                tmp += reg[k] * B[k * nq + r];
            }
            w1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
        }
    }
    // step 4 with JxW stored transposed per element, JxW_t(e, r, q, p) =
    // JxW(e, p, q, r): neighbouring lanes (p) read neighbouring values
    static void s4t(const int tid, const T* __restrict__ B, const T* __restrict__ JxW_t, const T* __restrict__ w0, T* __restrict__ w1)
    {
        const int e = tid / (nq * nq);
        const int q = tid % (nq * nq) / nq;
        const int p = tid % nq;
        T reg[nm];
        UNROLL
        for (int k = 0; k < nm; ++k) {
            reg[k] = w0[e * nq * nq * nm + q * nq * nm + p * nm + k];
        }
        UNROLL
        for (int r = 0; r < nq; ++r) {
            T tmp = 0;
            UNROLL
            for (int k = 0; k < nm; ++k) {
                tmp += reg[k] * B[k * nq + r];
            }
            w1[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                tmp * JxW_t[e * nq * nq * nq + r * nq * nq + q * nq + p];
        }
    }
    // step 6 : direction 2 back, thread (e, q, p): w1(e,r,q,p) -> w0(e,k,q,p)
    static void s6(const int tid, const T* __restrict__ B, const T* __restrict__ w1, T* __restrict__ w0)
    {
        const int e = tid / (nq * nq);
        const int q = tid % (nq * nq) / nq;
        const int p = tid % nq;
        T reg[nq];
        UNROLL
        for (int r = 0; r < nq; ++r) {
            reg[r] = w1[e * nq * nq * nq + r * nq * nq + q * nq + p];
        }
        UNROLL
        for (int k = 0; k < nm; ++k) {
            T tmp = 0;
            UNROLL
            for (int r = 0; r < nq; ++r) {
                tmp += reg[r] * B[k * nq + r];
            }
            w0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
        }
    }
    // step 7 : direction 1 back, thread (e, k, p): w0(e,k,q,p) -> w1(e,k,j,p)
    static void s7(const int tid, const T* __restrict__ B, const T* __restrict__ w0, T* __restrict__ w1)
    {
        const int e = tid / (nm * nq);
        const int k = tid % (nm * nq) / nq;
        const int p = tid % nq;
        T reg[nq];
        UNROLL
        for (int q = 0; q < nq; ++q) {
            reg[q] = w0[e * nm * nq * nq + k * nq * nq + q * nq + p];
        }
        UNROLL
        for (int j = 0; j < nm; ++j) {
            T tmp = 0;
            UNROLL
            for (int q = 0; q < nq; ++q) {
                tmp += reg[q] * B[j * nq + q];
            }
            w1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
        }
    }
    // step 8 : direction 0 back, thread (e, j, k): w1(e,k,j,p) -> w0(e,i,j,k)
    static void s8(const int tid, const T* __restrict__ B, const T* __restrict__ w1, T* __restrict__ w0)
    {
        const int e = tid / (nm * nm);
        const int j = tid % (nm * nm) / nm;
        const int k = tid % nm;
        T reg[nq];
        UNROLL
        for (int p = 0; p < nq; ++p) {
            reg[p] = w1[e * nm * nm * nq + k * nm * nq + j * nq + p];
        }
        UNROLL
        for (int i = 0; i < nm; ++i) {
            T tmp = 0;
            UNROLL
            for (int p = 0; p < nq; ++p) {
                tmp += reg[p] * B[i * nq + p];
            }
            w0[e * nm * nm * nm + i * nm * nm + j * nm + k] = tmp;
        }
    }
};

} // namespace bk1gpu

#endif // BK1_GPU_STEPS_H
