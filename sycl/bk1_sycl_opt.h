// bk1_sycl_opt.h -- the BK1 seven-step kernel "opt" of sycl/bk1_sycl.cpp,
// with its shape (nq, batch, work-group size, SLM layout) in one struct,
// opt_params, that comes either from template arguments (opt_kernel with
// fixed_shape: one order per binary, ahead of time) or from a
// specialization constant (opt_kernel with sc_shape: any order and shape in
// one JIT-compiled binary, chosen at run time).  The kernel body is the
// same in both.
//
// opt: the inline seven-step kernel (inl) with the tips of the oneAPI GPU
// optimization guide: SLM boxes laid out by bk1gpu::slm_layout (pad: strides
// padded against bank conflicts, scripts/study/slm_pad_search.py), optional
// sub-group block copies of in and out (block_io, syclex::group_load /
// group_store), and the work-group size chosen by the launch (fill: a power
// of two, so its hardware threads divide an Xe-core's 128).

#ifndef BK1_SYCL_OPT_H
#define BK1_SYCL_OPT_H

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "bk1_gpu.h"

#ifndef MAXNQ
#define MAXNQ 10
#endif

namespace bk1opt {

namespace syclex = sycl::ext::oneapi::experimental;

// the shape of one launch: nq points per direction, batch elements and block
// work-items per work-group (block >= batch * nq * nq), L the SLM strides
struct opt_params {
    int nq;
    int batch;
    int block;
    bk1gpu::slm_layout L;
};

constexpr opt_params make_opt_params(const int nq, const int batch, const int block, const bool pad)
{
    return {nq, batch, block, pad ? bk1gpu::slm_padded(nq) : bk1gpu::slm_dense(nq)};
}

// floats of local memory per work-group
constexpr int opt_lmem(const opt_params& P)
{
    return P.batch * (bk1gpu::slm_w0(P.nq, P.L) + bk1gpu::slm_w1(P.L));
}

// a shape the kernel can run: one work-item per point of the widest step
constexpr bool opt_valid(const opt_params& P)
{
    return P.nq >= 2 && P.nq <= MAXNQ && P.batch >= 1 && P.batch * P.nq * P.nq <= P.block && P.block <= 1024;
}

// Shape: fixed_shape<...> (the shape a compile-time constant) or sc_shape
// (the shape from the specialization constant opt_id, set by launch_sc)
template <int nq, int batch, int block, bool pad>
struct fixed_shape {
    static constexpr opt_params P = make_opt_params(nq, batch, block, pad);
    static_assert(opt_valid(P), "one work-item per point of the widest step, at most 1024");
};
struct sc_shape {};

inline constexpr sycl::specialization_id<opt_params> opt_id(make_opt_params(4, 1, 16, false));

// The kernel functor.  The pointers are its own members, not those of a
// nested struct: kernel_args_restrict then reaches them (with the pointers in
// a member struct the plain-copy variants ran 20% slower, p = 4, A770).  The
// block copies (group_load/store with full_group) need complete sub-groups:
// the group size is a multiple of 16, so the block_io kernels require SIMD16.
template <typename T, typename Shape, bool block_io>
struct opt_kernel {
    static constexpr bool whole_sub_groups()
    {
        if constexpr (block_io && !std::is_same_v<Shape, sc_shape>) {
            return Shape::P.block % 16 == 0;
        } else {
            return true;
        }
    }
    static_assert(whole_sub_groups(), "whole sub-groups of 16");
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    auto get(syclex::properties_tag) const
    {
        if constexpr (block_io) {
            return syclex::properties{syclex::sub_group_size<16>};
        } else {
            return syclex::properties{};
        }
    }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it, sycl::kernel_handler kh) const
    {
        if constexpr (std::is_same_v<Shape, sc_shape>) {
            run(it, kh.get_specialization_constant<opt_id>());
        } else {
            run(it, Shape::P);
        }
    }

    // P is a constant at every call (constexpr, or a specialization constant
    // folded by the JIT); run is inlined into the kernel
    void run(sycl::nd_item<1> it, const opt_params P) const
    {
        const int nq = P.nq;
        const int batch = P.batch;
        const bk1gpu::slm_layout L = P.L;
        const int nw0 = bk1gpu::slm_w0(nq, L);
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

// one launch of the run-time-shaped kernel over nelmt elements; the
// specialization constant selects (and on first use JIT-compiles) the shape
template <typename T, bool block_io>
sycl::event launch_sc(sycl::queue& q, const opt_params& P, const T* basis, const T* JxW_t, const T* in, T* out,
                      const int nelmt)
{
    const int ngroups = (nelmt + P.batch - 1) / P.batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * P.block, std::size_t(P.block)};
    return q.submit([&](sycl::handler& cgh) {
        cgh.set_specialization_constant<opt_id>(P);
        sycl::local_accessor<T, 1> lm(sycl::range<1>(opt_lmem(P)), cgh);
        cgh.parallel_for(range, opt_kernel<T, sc_shape, block_io>{basis, JxW_t, in, out, nelmt, lm});
    });
}

} // namespace bk1opt

#endif // BK1_SYCL_OPT_H
