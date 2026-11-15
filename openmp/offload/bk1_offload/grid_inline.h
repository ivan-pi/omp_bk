// grid_inline.h -- gsI: the seven-step kernel (gsT: JxW stored transposed
// per element, basis from global memory, contiguous copies of in and out)
// with every step's body written out in the kernel against the team arrays,
// instead of calling the gs_steps.h helpers.  In SYCL the same change took
// the kernel from ~320 to ~420 GB/s (device time): through the helpers'
// generic pointers the work boxes lost their local address space and the
// indexing became 64-bit (addc/mach in the assembly).
// Part of the BK1 OpenMP offload harness (openmp/offload/bk1_offload.cpp).

#ifndef BK1_OFFLOAD_GRID_INLINE_H
#define BK1_OFFLOAD_GRID_INLINE_H

#include <omp.h>

#include <cstddef>

#include "bk1_gpu.h"

namespace bk1gpu {

template <typename T, int nq, int batch, int block>
void GridStrideInline(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW_t,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW_t[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(block)
    for (int eb = 0; eb < nbatches; ++eb) {
        T w0[batch * nq3];
        T w1[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW_t + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                w0[i] = b_in[i];
            }
            // step 2 : thread (e, j, k): w0(e,i,j,k) -> w1(e,p,j,k)
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                const int e = t / (nm * nm);
                const int j = t % (nm * nm) / nm;
                const int k = t % nm;
                T reg[nm];
                UNROLL
                for (int i = 0; i < nm; ++i) {
                    reg[i] = w0[e * nm3 + i * nm * nm + j * nm + k];
                }
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    UNROLL
                    for (int i = 0; i < nm; ++i) {
                        tmp += reg[i] * basis[i * nq + p];
                    }
                    w1[e * nq * nm * nm + p * nm * nm + j * nm + k] = tmp;
                }
            }
            // step 3 : thread (e, p, k): w1(e,p,j,k) -> w0(e,q,p,k)
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                const int e = t / (nq * nm);
                const int p = t % (nq * nm) / nm;
                const int k = t % nm;
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
                        tmp += reg[j] * basis[j * nq + q];
                    }
                    w0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
                }
            }
            // step 4 : thread (e, q, p): w0(e,q,p,k) -> w1(e,r,q,p) * JxW_t(e,r,q,p)
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                const int e = t / (nq * nq);
                const int q = t % (nq * nq) / nq;
                const int p = t % nq;
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
                        tmp += reg[k] * basis[k * nq + r];
                    }
                    w1[e * nq3 + r * nq * nq + q * nq + p] = tmp * b_JxW[e * nq3 + r * nq * nq + q * nq + p];
                }
            }
            // step 6 : thread (e, q, p): w1(e,r,q,p) -> w0(e,k,q,p)
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                const int e = t / (nq * nq);
                const int q = t % (nq * nq) / nq;
                const int p = t % nq;
                T reg[nq];
                UNROLL
                for (int r = 0; r < nq; ++r) {
                    reg[r] = w1[e * nq3 + r * nq * nq + q * nq + p];
                }
                UNROLL
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    UNROLL
                    for (int r = 0; r < nq; ++r) {
                        tmp += reg[r] * basis[k * nq + r];
                    }
                    w0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
                }
            }
            // step 7 : thread (e, k, p): w0(e,k,q,p) -> w1(e,k,j,p)
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                const int e = t / (nm * nq);
                const int k = t % (nm * nq) / nq;
                const int p = t % nq;
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
                        tmp += reg[q] * basis[j * nq + q];
                    }
                    w1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
                }
            }
            // step 8 : thread (e, j, k): w1(e,k,j,p) -> w0(e,i,j,k)
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                const int e = t / (nm * nm);
                const int j = t % (nm * nm) / nm;
                const int k = t % nm;
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
                        tmp += reg[p] * basis[i * nq + p];
                    }
                    w0[e * nm3 + i * nm * nm + j * nm + k] = tmp;
                }
            }
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                b_out[i] = w0[i];
            }
        }
    }
}

// gsP: gsI with the SLM boxes laid out by slm_layout (common.h): padded
// strides against bank conflicts (pad = true) or the dense layout (pad =
// false, the same code as gsI), and a team of `block` work-items (may exceed
// batch * nq * nq, e.g. a power of two for the guide's occupancy rule).
template <typename T, int nq, int batch, int block, bool pad>
void GridStridePadded(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW_t,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    static_assert(batch * nq * nq <= block, "one work-item per point of the widest step");
    constexpr slm_layout L = pad ? slm_padded(nq) : slm_dense(nq);
    constexpr int nw0 = slm_w0(nq, L);
    constexpr int nw1 = slm_w1(L);
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW_t[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(block)
    for (int eb = 0; eb < nbatches; ++eb) {
        T w0[batch * nw0];
        T w1[batch * nw1];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW_t + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                w0[i] = b_in[i];
            }
            // step 2 : thread (e, j, k): w0(e,i,j,k) -> w1(e,p,j,k)
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                const int e = t / (nm * nm);
                const int j = t % (nm * nm) / nm;
                const int k = t % nm;
                T reg[nm];
                UNROLL
                for (int i = 0; i < nm; ++i) {
                    reg[i] = w0[e * nm3 + i * nm * nm + j * nm + k];
                }
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    UNROLL
                    for (int i = 0; i < nm; ++i) {
                        tmp += reg[i] * basis[i * nq + p];
                    }
                    w1[e * L.A.e + p * L.A.a + j * L.A.b + k] = tmp;
                }
            }
            // step 3 : thread (e, p, k): w1(e,p,j,k) -> w0(e,q,p,k)
            #pragma omp for
            for (int t = 0; t < cb * nq * nm; ++t) {
                const int e = t / (nq * nm);
                const int p = t % (nq * nm) / nm;
                const int k = t % nm;
                T reg[nm];
                UNROLL
                for (int j = 0; j < nm; ++j) {
                    reg[j] = w1[e * L.A.e + p * L.A.a + j * L.A.b + k];
                }
                UNROLL
                for (int q = 0; q < nq; ++q) {
                    T tmp = 0;
                    UNROLL
                    for (int j = 0; j < nm; ++j) {
                        tmp += reg[j] * basis[j * nq + q];
                    }
                    w0[e * L.B.e + q * L.B.a + p * L.B.b + k] = tmp;
                }
            }
            // step 4 : thread (e, q, p): w0(e,q,p,k) -> w1(e,r,q,p) * JxW_t(e,r,q,p)
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                const int e = t / (nq * nq);
                const int q = t % (nq * nq) / nq;
                const int p = t % nq;
                T reg[nm];
                UNROLL
                for (int k = 0; k < nm; ++k) {
                    reg[k] = w0[e * L.B.e + q * L.B.a + p * L.B.b + k];
                }
                UNROLL
                for (int r = 0; r < nq; ++r) {
                    T tmp = 0;
                    UNROLL
                    for (int k = 0; k < nm; ++k) {
                        tmp += reg[k] * basis[k * nq + r];
                    }
                    w1[e * L.C.e + r * L.C.a + q * L.C.b + p] = tmp * b_JxW[e * nq3 + r * nq * nq + q * nq + p];
                }
            }
            // step 6 : thread (e, q, p): w1(e,r,q,p) -> w0(e,k,q,p)
            #pragma omp for
            for (int t = 0; t < cb * nq * nq; ++t) {
                const int e = t / (nq * nq);
                const int q = t % (nq * nq) / nq;
                const int p = t % nq;
                T reg[nq];
                UNROLL
                for (int r = 0; r < nq; ++r) {
                    reg[r] = w1[e * L.C.e + r * L.C.a + q * L.C.b + p];
                }
                UNROLL
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    UNROLL
                    for (int r = 0; r < nq; ++r) {
                        tmp += reg[r] * basis[k * nq + r];
                    }
                    w0[e * L.D.e + k * L.D.a + q * L.D.b + p] = tmp;
                }
            }
            // step 7 : thread (e, k, p): w0(e,k,q,p) -> w1(e,k,j,p)
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
                const int e = t / (nm * nq);
                const int k = t % (nm * nq) / nq;
                const int p = t % nq;
                T reg[nq];
                UNROLL
                for (int q = 0; q < nq; ++q) {
                    reg[q] = w0[e * L.D.e + k * L.D.a + q * L.D.b + p];
                }
                UNROLL
                for (int j = 0; j < nm; ++j) {
                    T tmp = 0;
                    UNROLL
                    for (int q = 0; q < nq; ++q) {
                        tmp += reg[q] * basis[j * nq + q];
                    }
                    w1[e * L.E.e + k * L.E.a + j * L.E.b + p] = tmp;
                }
            }
            // step 8 : thread (e, j, k): w1(e,k,j,p) -> w0(e,i,j,k)
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
                const int e = t / (nm * nm);
                const int j = t % (nm * nm) / nm;
                const int k = t % nm;
                T reg[nq];
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    reg[p] = w1[e * L.E.e + k * L.E.a + j * L.E.b + p];
                }
                UNROLL
                for (int i = 0; i < nm; ++i) {
                    T tmp = 0;
                    UNROLL
                    for (int p = 0; p < nq; ++p) {
                        tmp += reg[p] * basis[i * nq + p];
                    }
                    w0[e * nm3 + i * nm * nm + j * nm + k] = tmp;
                }
            }
            #pragma omp for
            for (int i = 0; i < cb * nm3; ++i) {
                b_out[i] = w0[i];
            }
        }
    }
}

} // namespace bk1gpu

#endif // BK1_OFFLOAD_GRID_INLINE_H
