// bk1_sycl.cpp -- BK1 (mass operator) in SYCL, variants of the seven-step
// structure of the CUDA/Kokkos kernels (include/bk1_gpu_steps.h): a batch of
// elements per work-group, contiguous copies of in and out, the contractions
// through local memory, the basis read from global memory and JxW stored
// transposed per element, JxW_t(e, r, q, p) = JxW(e, p, q, r).
//
//   scripts/build.sh bk1_sycl_<p>       (-> build/a770/bk1_sycl_<p>, nq = p + 2)
//   scripts/build.sh bk1_sycl_<p>_jit   (JIT, for spec)
//   scripts/run.sh bk1_sycl_<p> [nelmt] [ntests]
//   BK_KERNEL="C_l_sg0_b1 spec" ...   run a subset (default: all)
//
// Variant names <C|D>_<g|l>_sg<N>_b<B>:
//   C  copy in -> local memory and local memory -> out (8 barriers)
//   D  step 2 reads in and step 8 writes out directly (6 barriers)
//   g  sycl::group_barrier (fences all memory at work-group scope)
//   l  nd_item::barrier(local_space) (fences local memory only)
//   sg the required sub-group size (0: the compiler's choice)
//   b  elements per work-group
// elem: one element per work-item (range; elem_wg128: nd_range, 128 per group;
//   elem_sg16/32: the same with that sub-group size required; elem_v4: in
//   and out moved as float4 vectors where the block allows)
// spec: C_l with nq from a specialization constant (register arrays sized
// for MAXNQ, loop bounds known at JIT time); use the bk1_sycl_<p>_jit build for the
// constant to be folded (an AOT build emulates it as a run-time buffer).
//
// Data: pseudo-random in and JxW (bk::fill_random_test_data), basis
// cos(i * nq + p).  The check is against a double-precision CPU reference:
// the largest difference relative to the largest |reference| (float
// rounding gives ~1e-7; an indexing error gives ~1).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"
#include "bk1_gpu_steps.h"

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#ifndef NQ
#define NQ 6
#endif
#ifndef MAXNQ
#define MAXNQ 10
#endif

namespace {

using bk1gpu::gs_steps;
namespace syclex = sycl::ext::oneapi::experimental;

constexpr int round16(const int n) { return bk::round_up(n, 16); }

// ---------------------------------------------------------------------------
// the seven-step kernel, compile-time nq
// ---------------------------------------------------------------------------
template <typename T, int nq, int batch, bool copy, bool local_fence>
struct seven_step {
    static constexpr int nm = nq - 1;
    static constexpr int nq3 = nq * nq * nq;
    static constexpr int nm3 = nm * nm * nm;
    static constexpr int block = round16(batch * nq * nq);
    static constexpr int lmem = batch * (nq3 + nq3);

    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    void bar(const sycl::nd_item<1>& it) const
    {
        if constexpr (local_fence) {
            it.barrier(sycl::access::fence_space::local_space);
        } else {
            sycl::group_barrier(it.get_group());
        }
    }

    void run(const sycl::nd_item<1>& it) const
    {
        using S = gs_steps<T, nq>;
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const int eb = static_cast<int>(it.get_group(0));
        const int remaining = nelmt - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW_t + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);
        T* w0 = lm.template get_multi_ptr<sycl::access::decorated::no>().get();
        T* w1 = w0 + batch * nq3;

        const T* src2 = b_in;
        if constexpr (copy) {
            for (int i = lid; i < cb * nm3; i += lsz) {
                w0[i] = b_in[i];
            }
            bar(it);
            src2 = w0;
        }
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            S::s2(t, basis, src2, w1);
        }
        bar(it);
        for (int t = lid; t < cb * nq * nm; t += lsz) {
            S::s3(t, basis, w1, w0);
        }
        bar(it);
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            S::s4t(t, basis, b_JxW, w0, w1);
        }
        bar(it);
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            S::s6(t, basis, w1, w0);
        }
        bar(it);
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            S::s7(t, basis, w0, w1);
        }
        bar(it);
        if constexpr (copy) {
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                S::s8(t, basis, w1, w0);
            }
            bar(it);
            for (int i = lid; i < cb * nm3; i += lsz) {
                b_out[i] = w0[i];
            }
        } else {
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                S::s8(t, basis, w1, b_out);
            }
        }
    }
};

template <typename K, int sg>
struct with_sg {
    K k;
    [[sycl::reqd_sub_group_size(sg), intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k.run(it); }
};

template <typename K>
struct no_sg {
    K k;
    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k.run(it); }
};

// ---------------------------------------------------------------------------
// the same kernel (C_l) with nq from a specialization constant
// ---------------------------------------------------------------------------
constexpr sycl::specialization_id<int> nq_id(4);

// nqc > 0: nq is that compile-time constant instead (the same source, for
// telling the code shape apart from the specialization constant)
template <typename T, int batch, int nqc = 0>
struct seven_step_spec {
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it, sycl::kernel_handler kh) const
    {
        int nq_value = nqc;
        if constexpr (nqc == 0) {
            nq_value = kh.get_specialization_constant<nq_id>();
        }
        const int nq = nq_value;
        const int nm = nq - 1;
        const int nq3 = nq * nq * nq;
        const int nm3 = nm * nm * nm;
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const int eb = static_cast<int>(it.get_group(0));
        const int remaining = nelmt - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW_t + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);
        T* w0 = lm.template get_multi_ptr<sycl::access::decorated::no>().get();
        T* w1 = w0 + batch * nq3;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        T reg[MAXNQ];

        for (int i = lid; i < cb * nm3; i += lsz) {
            w0[i] = b_in[i];
        }
        bar();
        // step 2 : thread (e, j, k): w0(e,i,j,k) -> w1(e,p,j,k)
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const int e = t / (nm * nm);
            const int j = t % (nm * nm) / nm;
            const int k = t % nm;
            for (int i = 0; i < nm; ++i) {
                reg[i] = w0[e * nm3 + i * nm * nm + j * nm + k];
            }
            for (int p = 0; p < nq; ++p) {
                T tmp = 0;
                for (int i = 0; i < nm; ++i) {
                    tmp += reg[i] * basis[i * nq + p];
                }
                w1[e * nq * nm * nm + p * nm * nm + j * nm + k] = tmp;
            }
        }
        bar();
        // step 3 : thread (e, p, k): w1(e,p,j,k) -> w0(e,q,p,k)
        for (int t = lid; t < cb * nq * nm; t += lsz) {
            const int e = t / (nq * nm);
            const int p = t % (nq * nm) / nm;
            const int k = t % nm;
            for (int j = 0; j < nm; ++j) {
                reg[j] = w1[e * nq * nm * nm + p * nm * nm + j * nm + k];
            }
            for (int q = 0; q < nq; ++q) {
                T tmp = 0;
                for (int j = 0; j < nm; ++j) {
                    tmp += reg[j] * basis[j * nq + q];
                }
                w0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
            }
        }
        bar();
        // step 4 : thread (e, q, p): w0(e,q,p,k) -> w1(e,r,q,p) * JxW_t(e,r,q,p)
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            const int e = t / (nq * nq);
            const int q = t % (nq * nq) / nq;
            const int p = t % nq;
            for (int k = 0; k < nm; ++k) {
                reg[k] = w0[e * nq * nq * nm + q * nq * nm + p * nm + k];
            }
            for (int r = 0; r < nq; ++r) {
                T tmp = 0;
                for (int k = 0; k < nm; ++k) {
                    tmp += reg[k] * basis[k * nq + r];
                }
                w1[e * nq3 + r * nq * nq + q * nq + p] = tmp * b_JxW[e * nq3 + r * nq * nq + q * nq + p];
            }
        }
        bar();
        // step 6 : thread (e, q, p): w1(e,r,q,p) -> w0(e,k,q,p)
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            const int e = t / (nq * nq);
            const int q = t % (nq * nq) / nq;
            const int p = t % nq;
            for (int r = 0; r < nq; ++r) {
                reg[r] = w1[e * nq3 + r * nq * nq + q * nq + p];
            }
            for (int k = 0; k < nm; ++k) {
                T tmp = 0;
                for (int r = 0; r < nq; ++r) {
                    tmp += reg[r] * basis[k * nq + r];
                }
                w0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
            }
        }
        bar();
        // step 7 : thread (e, k, p): w0(e,k,q,p) -> w1(e,k,j,p)
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            const int e = t / (nm * nq);
            const int k = t % (nm * nq) / nq;
            const int p = t % nq;
            for (int q = 0; q < nq; ++q) {
                reg[q] = w0[e * nm * nq * nq + k * nq * nq + q * nq + p];
            }
            for (int j = 0; j < nm; ++j) {
                T tmp = 0;
                for (int q = 0; q < nq; ++q) {
                    tmp += reg[q] * basis[j * nq + q];
                }
                w1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
            }
        }
        bar();
        // step 8 : thread (e, j, k): w1(e,k,j,p) -> w0(e,i,j,k)
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const int e = t / (nm * nm);
            const int j = t % (nm * nm) / nm;
            const int k = t % nm;
            for (int p = 0; p < nq; ++p) {
                reg[p] = w1[e * nm * nm * nq + k * nm * nq + j * nq + p];
            }
            for (int i = 0; i < nm; ++i) {
                T tmp = 0;
                for (int p = 0; p < nq; ++p) {
                    tmp += reg[p] * basis[i * nq + p];
                }
                w0[e * nm3 + i * nm * nm + j * nm + k] = tmp;
            }
        }
        bar();
        for (int i = lid; i < cb * nm3; i += lsz) {
            b_out[i] = w0[i];
        }
    }
};

// ---------------------------------------------------------------------------
// opt: the inline seven-step kernel (inl) with the tips of the oneAPI GPU
// optimization guide: SLM boxes laid out by bk1gpu::slm_layout (pad: strides
// padded against bank conflicts, scripts/study/slm_pad_search.py), optional
// sub-group block copies of in and out (block_io, syclex::group_load /
// group_store), and the work-group size chosen by the launch (fill: a power
// of two, so its hardware threads divide an Xe-core's 128).
// ---------------------------------------------------------------------------
template <typename T, int nq_c, int batch, bool pad, bool block_io>
struct seven_step_opt {
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        constexpr int nq = nq_c;
        constexpr bk1gpu::slm_layout L = pad ? bk1gpu::slm_padded(nq) : bk1gpu::slm_dense(nq);
        constexpr int nw0 = bk1gpu::slm_w0(nq, L);
        const int nm = nq - 1;
        const int nq3 = nq * nq * nq;
        const int nm3 = nm * nm * nm;
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const int eb = static_cast<int>(it.get_group(0));
        const int remaining = nelmt - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_JxW = JxW_t + std::size_t(eb) * (batch * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);
        T* w0 = lm.template get_multi_ptr<sycl::access::decorated::no>().get();
        T* w1 = w0 + batch * nw0;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        T reg[MAXNQ];

        if constexpr (block_io) {
            // sub-group block loads: each sub-group moves chunks of 4 x its
            // size contiguous floats, one address per message
            const sycl::sub_group sg = it.get_sub_group();
            const int sgs = static_cast<int>(sg.get_local_linear_range());
            const int sgl = static_cast<int>(sg.get_local_linear_id());
            const int chunk = 4 * sgs;
            const int nchunks = cb * nm3 / chunk;
            for (int c = static_cast<int>(sg.get_group_linear_id()); c < nchunks;
                 c += static_cast<int>(sg.get_group_linear_range())) {
                // through uint32_t: in icpx 2026.1, group_load of float
                // converts the loaded bits as integer values (1.0f -> 1.065e9)
                static_assert(sizeof(T) == sizeof(uint32_t), "block copies move 32-bit values");
                sycl::vec<uint32_t, 4> v;
                syclex::group_load(sg, reinterpret_cast<const uint32_t*>(b_in + c * chunk), v,
                                   syclex::properties{syclex::data_placement_striped, syclex::contiguous_memory,
                                                      syclex::full_group});
                UNROLL
                for (int r = 0; r < 4; ++r) {
                    w0[c * chunk + r * sgs + sgl] = sycl::bit_cast<T>(uint32_t(v[r]));
                }
            }
            for (int i = nchunks * chunk + lid; i < cb * nm3; i += lsz) {
                w0[i] = b_in[i];
            }
        } else {
            for (int i = lid; i < cb * nm3; i += lsz) {
                w0[i] = b_in[i];
            }
        }
        bar();
        // step 2 : thread (e, j, k): w0(e,i,j,k) -> w1(e,p,j,k)
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const int e = t / (nm * nm);
            const int j = t % (nm * nm) / nm;
            const int k = t % nm;
            for (int i = 0; i < nm; ++i) {
                reg[i] = w0[e * nm3 + i * nm * nm + j * nm + k];
            }
            for (int p = 0; p < nq; ++p) {
                T tmp = 0;
                for (int i = 0; i < nm; ++i) {
                    tmp += reg[i] * basis[i * nq + p];
                }
                w1[e * L.A.e + p * L.A.a + j * L.A.b + k] = tmp;
            }
        }
        bar();
        // step 3 : thread (e, p, k): w1(e,p,j,k) -> w0(e,q,p,k)
        for (int t = lid; t < cb * nq * nm; t += lsz) {
            const int e = t / (nq * nm);
            const int p = t % (nq * nm) / nm;
            const int k = t % nm;
            for (int j = 0; j < nm; ++j) {
                reg[j] = w1[e * L.A.e + p * L.A.a + j * L.A.b + k];
            }
            for (int q = 0; q < nq; ++q) {
                T tmp = 0;
                for (int j = 0; j < nm; ++j) {
                    tmp += reg[j] * basis[j * nq + q];
                }
                w0[e * L.B.e + q * L.B.a + p * L.B.b + k] = tmp;
            }
        }
        bar();
        // step 4 : thread (e, q, p): w0(e,q,p,k) -> w1(e,r,q,p) * JxW_t(e,r,q,p)
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            const int e = t / (nq * nq);
            const int q = t % (nq * nq) / nq;
            const int p = t % nq;
            for (int k = 0; k < nm; ++k) {
                reg[k] = w0[e * L.B.e + q * L.B.a + p * L.B.b + k];
            }
            for (int r = 0; r < nq; ++r) {
                T tmp = 0;
                for (int k = 0; k < nm; ++k) {
                    tmp += reg[k] * basis[k * nq + r];
                }
                w1[e * L.C.e + r * L.C.a + q * L.C.b + p] = tmp * b_JxW[e * nq3 + r * nq * nq + q * nq + p];
            }
        }
        bar();
        // step 6 : thread (e, q, p): w1(e,r,q,p) -> w0(e,k,q,p)
        for (int t = lid; t < cb * nq * nq; t += lsz) {
            const int e = t / (nq * nq);
            const int q = t % (nq * nq) / nq;
            const int p = t % nq;
            for (int r = 0; r < nq; ++r) {
                reg[r] = w1[e * L.C.e + r * L.C.a + q * L.C.b + p];
            }
            for (int k = 0; k < nm; ++k) {
                T tmp = 0;
                for (int r = 0; r < nq; ++r) {
                    tmp += reg[r] * basis[k * nq + r];
                }
                w0[e * L.D.e + k * L.D.a + q * L.D.b + p] = tmp;
            }
        }
        bar();
        // step 7 : thread (e, k, p): w0(e,k,q,p) -> w1(e,k,j,p)
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            const int e = t / (nm * nq);
            const int k = t % (nm * nq) / nq;
            const int p = t % nq;
            for (int q = 0; q < nq; ++q) {
                reg[q] = w0[e * L.D.e + k * L.D.a + q * L.D.b + p];
            }
            for (int j = 0; j < nm; ++j) {
                T tmp = 0;
                for (int q = 0; q < nq; ++q) {
                    tmp += reg[q] * basis[j * nq + q];
                }
                w1[e * L.E.e + k * L.E.a + j * L.E.b + p] = tmp;
            }
        }
        bar();
        // step 8 : thread (e, j, k): w1(e,k,j,p) -> w0(e,i,j,k)
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const int e = t / (nm * nm);
            const int j = t % (nm * nm) / nm;
            const int k = t % nm;
            for (int p = 0; p < nq; ++p) {
                reg[p] = w1[e * L.E.e + k * L.E.a + j * L.E.b + p];
            }
            for (int i = 0; i < nm; ++i) {
                T tmp = 0;
                for (int p = 0; p < nq; ++p) {
                    tmp += reg[p] * basis[i * nq + p];
                }
                w0[e * nm3 + i * nm * nm + j * nm + k] = tmp;
            }
        }
        bar();
        if constexpr (block_io) {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgs = static_cast<int>(sg.get_local_linear_range());
            const int sgl = static_cast<int>(sg.get_local_linear_id());
            const int chunk = 4 * sgs;
            const int nchunks = cb * nm3 / chunk;
            for (int c = static_cast<int>(sg.get_group_linear_id()); c < nchunks;
                 c += static_cast<int>(sg.get_group_linear_range())) {
                sycl::vec<uint32_t, 4> v;
                UNROLL
                for (int r = 0; r < 4; ++r) {
                    v[r] = sycl::bit_cast<uint32_t>(w0[c * chunk + r * sgs + sgl]);
                }
                syclex::group_store(sg, v, reinterpret_cast<uint32_t*>(b_out + c * chunk),
                                    syclex::properties{syclex::data_placement_striped, syclex::contiguous_memory,
                                                       syclex::full_group});
            }
            for (int i = nchunks * chunk + lid; i < cb * nm3; i += lsz) {
                b_out[i] = w0[i];
            }
        } else {
            for (int i = lid; i < cb * nm3; i += lsz) {
                b_out[i] = w0[i];
            }
        }
    }
};


// ---------------------------------------------------------------------------
// one element per work-item (the reference SumFactorization's structure, as
// the OpenMP `teams loop` kernel): private work arrays, no local memory and
// no barriers; the lanes of a sub-group run across elements.  Fast while the
// 2 * nq^3 private floats stay in registers (p = 1, 2); from p = 4 they spill.
// JxW is the transposed per-element layout the harness passes to all
// kernels, JxW_t(e, r, q, p).
// ---------------------------------------------------------------------------
template <typename T, int nq, bool vec_io = false>
struct one_element {
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;

    void run(const int e) const
    {
        constexpr int nm = nq - 1;
        constexpr int nq3 = nq * nq * nq;
        constexpr int nm3 = nm * nm * nm;
        if (e >= nelmt) {
            return;
        }
        const T* ei = in + std::size_t(e) * nm3;
        const T* ej = JxW_t + std::size_t(e) * nq3;
        T* eo = out + std::size_t(e) * nm3;
        T w0[nq3];
        T w1[nq3];
        // the element's input into private memory first (one global load per
        // value, as the reference's step 1), then direction 0:
        // w1(p, j, k) = sum_i in(i, j, k) B(i, p)
        T u[nm3];
        if constexpr (vec_io && nm3 % 4 == 0) {
            // the element's block is nm3 floats, a multiple of 16 bytes: load it
            // as float4 vectors (one 4-wide message per lane instead of four)
            UNROLL
            for (int i = 0; i < nm3; i += 4) {
                const sycl::vec<T, 4> v = *reinterpret_cast<const sycl::vec<T, 4>*>(ei + i);
                u[i] = v[0];
                u[i + 1] = v[1];
                u[i + 2] = v[2];
                u[i + 3] = v[3];
            }
        } else {
            UNROLL
            for (int i = 0; i < nm3; ++i) {
                u[i] = ei[i];
            }
        }
        UNROLL
        for (int p = 0; p < nq; ++p) {
            UNROLL
            for (int k = 0; k < nm; ++k) {
                UNROLL
                for (int j = 0; j < nm; ++j) {
                    T tmp = 0;
                    UNROLL
                    for (int i = 0; i < nm; ++i) {
                        tmp += u[(i * nm + j) * nm + k] * basis[i * nq + p];
                    }
                    w1[(p * nq + j) * nq + k] = tmp;
                }
            }
        }
        // direction 1: w0(q, p, k) = sum_j w1(p, j, k) B(j, q)
        UNROLL
        for (int q = 0; q < nq; ++q) {
            UNROLL
            for (int p = 0; p < nq; ++p) {
                UNROLL
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    UNROLL
                    for (int j = 0; j < nm; ++j) {
                        tmp += w1[(p * nq + j) * nq + k] * basis[j * nq + q];
                    }
                    w0[(q * nq + p) * nq + k] = tmp;
                }
            }
        }
        // direction 2 and JxW: w1(p, q, r) = sum_k w0(q, p, k) B(k, r) * JxW(p, q, r)
        UNROLL
        for (int r = 0; r < nq; ++r) {
            UNROLL
            for (int q = 0; q < nq; ++q) {
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    UNROLL
                    for (int k = 0; k < nm; ++k) {
                        tmp += w0[(q * nq + p) * nq + k] * basis[k * nq + r];
                    }
                    w1[(p * nq + q) * nq + r] = tmp * ej[(r * nq + q) * nq + p];
                }
            }
        }
        // direction 2 back: w0(q, p, k) = sum_r w1(p, q, r) B(k, r)
        UNROLL
        for (int k = 0; k < nm; ++k) {
            UNROLL
            for (int q = 0; q < nq; ++q) {
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    UNROLL
                    for (int r = 0; r < nq; ++r) {
                        tmp += w1[(p * nq + q) * nq + r] * basis[k * nq + r];
                    }
                    w0[(q * nq + p) * nq + k] = tmp;
                }
            }
        }
        // direction 1 back: w1(p, j, k) = sum_q w0(q, p, k) B(j, q)
        UNROLL
        for (int j = 0; j < nm; ++j) {
            UNROLL
            for (int k = 0; k < nm; ++k) {
                UNROLL
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    UNROLL
                    for (int q = 0; q < nq; ++q) {
                        tmp += w0[(q * nq + p) * nq + k] * basis[j * nq + q];
                    }
                    w1[(p * nq + j) * nq + k] = tmp;
                }
            }
        }
        // direction 0 back: out(i, j, k) = sum_p w1(p, j, k) B(i, p)
        T o[nm3];
        UNROLL
        for (int i = 0; i < nm; ++i) {
            UNROLL
            for (int j = 0; j < nm; ++j) {
                UNROLL
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    UNROLL
                    for (int p = 0; p < nq; ++p) {
                        tmp += w1[(p * nq + j) * nq + k] * basis[i * nq + p];
                    }
                    o[(i * nm + j) * nm + k] = tmp;
                }
            }
        }
        if constexpr (vec_io && nm3 % 4 == 0) {
            UNROLL
            for (int i = 0; i < nm3; i += 4) {
                *reinterpret_cast<sycl::vec<T, 4>*>(eo + i) = sycl::vec<T, 4>(o[i], o[i + 1], o[i + 2], o[i + 3]);
            }
        } else {
            UNROLL
            for (int i = 0; i < nm3; ++i) {
                eo[i] = o[i];
            }
        }
    }
};

template <typename K>
struct elem_range {
    K k;
    [[intel::kernel_args_restrict]] void operator()(sycl::item<1> it) const { k.run(static_cast<int>(it.get_id(0))); }
};

template <typename K, int sg>
struct elem_nd_sg {
    K k;
    [[sycl::reqd_sub_group_size(sg), intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        k.run(static_cast<int>(it.get_global_id(0)));
    }
};

template <typename K>
struct elem_nd {
    K k;
    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k.run(static_cast<int>(it.get_global_id(0))); }
};

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
struct device_data {
    float* basis;
    float* JxW_t;
    float* in;
    float* out;
    int nelmt;
};

using launch_fn = sycl::event (*)(sycl::queue&, const device_data&);

template <int nq, int batch, bool copy, bool local_fence, int sg>
sycl::event launch(sycl::queue& q, const device_data& d)
{
    using K = seven_step<float, nq, batch, copy, local_fence>;
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * K::block, std::size_t(K::block)};
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>(K::lmem), cgh);
        const K k{d.basis, d.JxW_t, d.in, d.out, d.nelmt, lm};
        if constexpr (sg > 0) {
            cgh.parallel_for(range, with_sg<K, sg>{k});
        } else {
            cgh.parallel_for(range, no_sg<K>{k});
        }
    });
}

template <int batch, int nqc = 0>
sycl::event launch_spec(sycl::queue& q, const device_data& d)
{
    constexpr int nq = NQ;
    const int block = round16(batch * nq * nq);
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * block, std::size_t(block)};
    return q.submit([&](sycl::handler& cgh) {
        if constexpr (nqc == 0) {
            cgh.set_specialization_constant<nq_id>(nq);
        }
        sycl::local_accessor<float, 1> lm(sycl::range<1>(batch * 2 * nq * nq * nq), cgh);
        cgh.parallel_for(range, seven_step_spec<float, batch, nqc>{d.basis, d.JxW_t, d.in, d.out, d.nelmt, lm});
    });
}

// the block copies (group_load/store with full_group) need complete
// sub-groups: the group size is a multiple of 16, so they require SIMD16
template <typename K>
struct sg16 {
    K k;
    [[sycl::reqd_sub_group_size(16), intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k(it); }
};

template <int batch, int block, bool pad, bool block_io>
sycl::event launch_opt(sycl::queue& q, const device_data& d)
{
    constexpr int nq = NQ;
    static_assert(batch * nq * nq <= block && block <= 1024, "one work-item per point of the widest step");
    constexpr bk1gpu::slm_layout L = pad ? bk1gpu::slm_padded(nq) : bk1gpu::slm_dense(nq);
    constexpr int lmem = batch * (bk1gpu::slm_w0(nq, L) + bk1gpu::slm_w1(L));
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * block, std::size_t(block)};
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>(lmem), cgh);
        using K = seven_step_opt<float, nq, batch, pad, block_io>;
        const K k{d.basis, d.JxW_t, d.in, d.out, d.nelmt, lm};
        if constexpr (block_io) {
            static_assert(block % 16 == 0, "whole sub-groups of 16");
            cgh.parallel_for(range, sg16<K>{k});
        } else {
            cgh.parallel_for(range, k);
        }
    });
}

template <int wg, int sg = 0, bool vec_io = false>
sycl::event launch_elem(sycl::queue& q, const device_data& d)
{
    using K = one_element<float, NQ, vec_io>;
    const K k{d.basis, d.JxW_t, d.in, d.out, d.nelmt};
    if constexpr (wg == 0) {
        return q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(std::size_t(d.nelmt)), elem_range<K>{k});
        });
    } else {
        const std::size_t global = (std::size_t(d.nelmt) + wg - 1) / wg * wg;
        return q.submit([&](sycl::handler& cgh) {
            if constexpr (sg > 0) {
                cgh.parallel_for(sycl::nd_range<1>(global, wg), elem_nd_sg<K, sg>{k});
            } else {
                cgh.parallel_for(sycl::nd_range<1>(global, wg), elem_nd<K>{k});
            }
        });
    }
}

struct variant {
    const char* name;
    launch_fn fn;
};

#define V(C, G, SG, B) \
    {#C "_" #G "_sg" #SG "_b" #B, &launch<NQ, bk1gpu::cap_batch(NQ, B), (#C[0] == 'C'), (#G[0] == 'l'), SG>}

const variant variants[] = {
    V(C, g, 0, 1), V(C, l, 0, 1), V(D, g, 0, 1), V(D, l, 0, 1),
    V(C, g, 0, 4), V(C, l, 0, 4), V(D, g, 0, 4), V(D, l, 0, 4),
    V(C, l, 16, 1), V(C, l, 32, 1), V(C, l, 16, 4), V(C, l, 32, 4),
    V(D, l, 16, 1), V(D, l, 32, 1), V(D, l, 16, 4), V(D, l, 32, 4),
    V(C, l, 0, 8), V(D, l, 0, 8),
    {"spec_b1", &launch_spec<1>},
    {"spec_b4", &launch_spec<bk1gpu::cap_batch(NQ, 4)>},
    {"inl_b1", &launch_spec<1, NQ>},
    {"inl_b4", &launch_spec<bk1gpu::cap_batch(NQ, 4), NQ>},
    {"inl_b8", &launch_spec<bk1gpu::cap_batch(NQ, 8), NQ>},
    {"inl_b16", &launch_spec<bk1gpu::cap_batch(NQ, 16), NQ>},
    {"inlP_b1", &launch_opt<bk1gpu::cap_batch(NQ, 1), round16(bk1gpu::cap_batch(NQ, 1) * NQ * NQ), true, false>},
    {"inlP_b4", &launch_opt<bk1gpu::cap_batch(NQ, 4), round16(bk1gpu::cap_batch(NQ, 4) * NQ * NQ), true, false>},
    {"inlP_b8", &launch_opt<bk1gpu::cap_batch(NQ, 8), round16(bk1gpu::cap_batch(NQ, 8) * NQ * NQ), true, false>},
    {"inlP_b16", &launch_opt<bk1gpu::cap_batch(NQ, 16), round16(bk1gpu::cap_batch(NQ, 16) * NQ * NQ), true, false>},
    {"inlB_b1", &launch_opt<bk1gpu::cap_batch(NQ, 1), round16(bk1gpu::cap_batch(NQ, 1) * NQ * NQ), false, true>},
    {"inlBP_b1", &launch_opt<bk1gpu::cap_batch(NQ, 1), round16(bk1gpu::cap_batch(NQ, 1) * NQ * NQ), true, true>},
    {"inlB_b4", &launch_opt<bk1gpu::cap_batch(NQ, 4), round16(bk1gpu::cap_batch(NQ, 4) * NQ * NQ), false, true>},
    {"inlBP_b4", &launch_opt<bk1gpu::cap_batch(NQ, 4), round16(bk1gpu::cap_batch(NQ, 4) * NQ * NQ), true, true>},
    {"inlB_b16", &launch_opt<bk1gpu::cap_batch(NQ, 16), round16(bk1gpu::cap_batch(NQ, 16) * NQ * NQ), false, true>},
    {"inlBP_b16", &launch_opt<bk1gpu::cap_batch(NQ, 16), round16(bk1gpu::cap_batch(NQ, 16) * NQ * NQ), true, true>},
    {"inlF_f128", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 128)), 128, false, false>},
    {"inlFP_f128", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 128)), 128, true, false>},
    {"inlF_f256", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 256)), 256, false, false>},
    {"inlFP_f256", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 256)), 256, true, false>},
    {"inlF_f512", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 512)), 512, false, false>},
    {"inlFP_f512", &launch_opt<bk1gpu::cap_batch(NQ, bk1gpu::fill_batch(NQ, 512)), 512, true, false>},
#ifndef BK_NO_BASELINES
    // one element per work-item: the naive baselines, 100x slower than the
    // batched kernels on the A770 and long enough at high order to trip a
    // small GPU's hang check (UHD 750, p = 6); -DBK_NO_BASELINES leaves them out
    {"elem", &launch_elem<0>},
    {"elem_wg128", &launch_elem<128>},
    {"elem_sg16", &launch_elem<128, 16>},
    {"elem_sg32", &launch_elem<128, 32>},
    {"elem_v4", &launch_elem<128, 0, true>},
#endif
    {"inl_b32", &launch_spec<bk1gpu::cap_batch(NQ, 32), NQ>},
    {"inl_b64", &launch_spec<bk1gpu::cap_batch(NQ, 64), NQ>},
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 19;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 20;

    // BK_PROFILE=0: no event profiling (no device times, less launch
    // overhead); BK_INORDER=1: an in-order queue
    const bool profile = bk::get_env("BK_PROFILE").value_or("1") != "0";
    const bool in_order = bk::get_env("BK_INORDER").value_or("0") == "1";
    sycl::property_list props;
    if (profile && in_order) {
        props = sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}};
    } else if (profile) {
        props = sycl::property_list{sycl::property::queue::enable_profiling{}};
    } else if (in_order) {
        props = sycl::property_list{sycl::property::queue::in_order{}};
    }
    sycl::queue q{sycl::gpu_selector_v, props};
    std::printf("BK1 SYCL: p = %d, nq = %d, nelmt = %zu, ntests = %d, device %s\n", nq - 2, nq, nelmt,
                ntests, q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    std::vector<float> JxW(nelmt * nq * nq * nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    bk::fill_random_test_data(in, JxW);
    const std::vector<float> JxW_t = bk::transpose_jxw(JxW, nelmt, nq);
    std::vector<double> ref(in.size());
    bk::reference_bk1(nq, nelmt, basis, JxW, in, ref);
    const double big = bk::max_abs(ref);

    device_data d{sycl::malloc_device<float>(basis.size(), q), sycl::malloc_device<float>(JxW_t.size(), q),
                  sycl::malloc_device<float>(in.size(), q), sycl::malloc_device<float>(in.size(), q),
                  static_cast<int>(nelmt)};
    q.copy(basis.data(), d.basis, basis.size());
    q.copy(JxW_t.data(), d.JxW_t, JxW_t.size());
    q.copy(in.data(), d.in, in.size());
    q.wait_and_throw();

    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());
    std::vector<float> out(in.size());
    for (const variant& v : variants) {
        if (!bk::selected(v.name)) {
            continue;
        }
        q.memset(d.out, 0, out.size() * sizeof(float)).wait();
        bk::launch_times t{};
        if (!bk::time_variant(v.name, ntests, [&] { return v.fn(q, d); }, t, profile)) {
            continue;
        }
        q.copy(d.out, out.data(), out.size()).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s\n",
                    v.name, 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host,
                    1e-9 * in.size() / t.device, 1e-9 * bytes / t.device, relerr, bk::fail_mark(relerr));
    }
    sycl::free(d.basis, q);
    sycl::free(d.JxW_t, q);
    sycl::free(d.in, q);
    sycl::free(d.out, q);
    return 0;
}
