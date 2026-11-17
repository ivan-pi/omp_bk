// bk1_sycl_opt.h -- the BK1 seven-step kernel "opt" of the SYCL drivers,
// with its shape (nq, batch, work-group size, SLM layout) in one struct,
// opt_params.  The shape comes from template arguments (fixed_shape: one
// order per binary, sycl/bk1_sycl.cpp) or from the specialization constant
// opt_id (sc_shape: any order and shape in one JIT-compiled binary,
// sycl/bk1_sycl_sc.cpp); the kernel body is the same.  opt_variants is the
// table of shapes both drivers run.
//
// opt: the seven-step kernel with the tips of the oneAPI GPU optimization
// guide: SLM boxes laid out by bk1gpu::slm_layout (dense, or padded against
// bank conflicts, scripts/study/slm_pad_search.py), optional sub-group block
// copies of in and out (block_io, syclex::group_load / group_store), and the
// work-group size chosen by the launch (fill: a power of two, so its
// hardware threads divide an Xe-core's 128).

#ifndef BK1_SYCL_OPT_H
#define BK1_SYCL_OPT_H

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <cstddef>
#include <cstdint>

#include "bk1_gpu.h"
#include "bk_sycl.h"

#ifndef MAXNQ
#define MAXNQ 10
#endif

namespace bk1opt {

namespace syclex = sycl::ext::oneapi::experimental;

// the shape of one launch: nq points per direction, batch elements and block
// work-items per work-group, L the SLM strides
struct opt_params {
    int nq;
    int batch;
    int block;
    int slm;   // slm_kind of L
    bk1gpu::slm_layout L;
};

// the SLM layout of the boxes (bk1gpu::slm_dense, slm_padded)
enum slm_kind : int { dense = 0, padded = 1 };

constexpr opt_params make_opt_params(const int nq, const int batch, const int block, const int slm)
{
    return {nq, batch, block, slm, (slm == padded) ? bk1gpu::slm_padded(nq) : bk1gpu::slm_dense(nq)};
}

constexpr bool same_shape(const opt_params& a, const opt_params& b)
{
    return a.nq == b.nq && a.batch == b.batch && a.block == b.block && a.slm == b.slm;
}

// floats of local memory per work-group
constexpr int opt_lmem(const opt_params& P)
{
    return P.batch * (bk1gpu::slm_w0(P.nq, P.L) + bk1gpu::slm_w1(P.L));
}

// the launch geometry of shape P (include/bk_sycl.h); the block copies need
// SIMD16
constexpr bk::kernel_shape opt_shape(const opt_params& P, const bool block_io)
{
    return {P.batch, P.block, block_io ? 16 : 0, 1, opt_lmem(P)};
}

// a shape the kernel can run: one work-item per point of the widest step,
// within the device's limits (bk::shape_valid)
constexpr bool opt_valid(const opt_params& P, const bool block_io)
{
    return P.nq >= 2 && P.nq <= MAXNQ && P.batch * P.nq * P.nq <= P.block && bk::shape_valid(opt_shape(P, block_io));
}

// The shapes of the drivers, named <aot> in bk1_sycl.cpp and <sc> in
// bk1_sycl_sc.cpp: `want` elements per work-group (capped by
// bk1gpu::cap_batch) and a group of whole sub-groups over the widest step,
// or (threads > 0) a group of `threads` filled with elements.
struct opt_variant {
    const char* aot;
    const char* sc;
    int want;
    int threads;
    int slm;   // slm_kind
    bool block_io;
};

inline constexpr opt_variant opt_variants[] = {
    {"inl_b1", "sc_b1", 1, 0, dense, false},
    {"inl_b4", "sc_b4", 4, 0, dense, false},
    {"inl_b8", "sc_b8", 8, 0, dense, false},
    {"inl_b16", "sc_b16", 16, 0, dense, false},
    {"inl_b32", "sc_b32", 32, 0, dense, false},
    {"inl_b64", "sc_b64", 64, 0, dense, false},
    {"inlP_b1", "scP_b1", 1, 0, padded, false},
    {"inlP_b4", "scP_b4", 4, 0, padded, false},
    {"inlP_b8", "scP_b8", 8, 0, padded, false},
    {"inlP_b16", "scP_b16", 16, 0, padded, false},
    {"inlB_b1", "scB_b1", 1, 0, dense, true},
    {"inlBP_b1", "scBP_b1", 1, 0, padded, true},
    {"inlB_b4", "scB_b4", 4, 0, dense, true},
    {"inlBP_b4", "scBP_b4", 4, 0, padded, true},
    {"inlB_b16", "scB_b16", 16, 0, dense, true},
    {"inlBP_b16", "scBP_b16", 16, 0, padded, true},
    {"inlF_f128", "scF_f128", 0, 128, dense, false},
    {"inlFP_f128", "scFP_f128", 0, 128, padded, false},
    {"inlF_f256", "scF_f256", 0, 256, dense, false},
    {"inlFP_f256", "scFP_f256", 0, 256, padded, false},
    {"inlF_f512", "scF_f512", 0, 512, dense, false},
    {"inlFP_f512", "scFP_f512", 0, 512, padded, false},
};

constexpr opt_params variant_params(const int nq, const opt_variant& v)
{
    const int batch = bk1gpu::cap_batch(nq, (v.threads > 0) ? bk1gpu::fill_batch(nq, v.threads) : v.want);
    const int block = (v.threads > 0) ? v.threads : bk1gpu::gs_block(nq, batch);
    return make_opt_params(nq, batch, block, v.slm);
}

// Shape: where the kernel gets its opt_params.  set() runs at submission,
// get() in the kernel.
inline constexpr sycl::specialization_id<opt_params> opt_id(make_opt_params(4, 1, 16, dense));

template <int nq, int batch, int block, int slm>
struct fixed_shape {
    static constexpr opt_params P = make_opt_params(nq, batch, block, slm);
    static void set(sycl::handler&, const opt_params&) {}
    static opt_params get(sycl::kernel_handler&) { return P; }
};

struct sc_shape {
    static void set(sycl::handler& cgh, const opt_params& P) { cgh.set_specialization_constant<opt_id>(P); }
    static opt_params get(sycl::kernel_handler& kh) { return kh.get_specialization_constant<opt_id>(); }
};

template <typename T>
struct device_data {
    T* basis;
    T* JxW_t;   // JxW transposed per element, (e, r, q, p)
    T* in;
    T* out;
    int nelmt;
};

// The kernel functor, launched itself, never through a wrapper functor
// (include/bk_sycl.h).  The block copies need SIMD16.
template <typename T, typename Shape, bool block_io>
struct opt_kernel {
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    auto get(syclex::properties_tag) const { return bk::launch_properties<block_io ? 16 : 0>(); }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it, sycl::kernel_handler kh) const
    {
        run(it, Shape::get(kh));
    }

    // dst[0, n) = src[0, n) by the whole work-group, load: global src to
    // local dst, else local src to global dst.  With block_io, the global
    // side moves in sub-group block messages of 4 x the sub-group size
    // contiguous values (one address per message), the rest one value per
    // work-item.  The block messages go through uint32_t: in icpx 2026.1,
    // group_load of float converts the loaded bits as integer values
    // (1.0f -> 1.065e9).
    template <bool load>
    static void group_copy(const sycl::nd_item<1>& it, const T* src, T* dst, const int n)
    {
        int done = 0;
        if constexpr (block_io) {
            static_assert(sizeof(T) == sizeof(uint32_t), "block copies move 32-bit values");
            constexpr auto props =
                syclex::properties{syclex::data_placement_striped, syclex::contiguous_memory, syclex::full_group};
            const sycl::sub_group sg = it.get_sub_group();
            const int sgs = static_cast<int>(sg.get_local_linear_range());
            const int sgl = static_cast<int>(sg.get_local_linear_id());
            const int chunk = 4 * sgs;
            const int nchunks = n / chunk;
            for (int c = static_cast<int>(sg.get_group_linear_id()); c < nchunks;
                 c += static_cast<int>(sg.get_group_linear_range())) {
                sycl::vec<uint32_t, 4> v;
                if constexpr (load) {
                    syclex::group_load(sg, reinterpret_cast<const uint32_t*>(src + c * chunk), v, props);
                    UNROLL
                    for (int r = 0; r < 4; ++r) {
                        dst[c * chunk + r * sgs + sgl] = sycl::bit_cast<T>(uint32_t(v[r]));
                    }
                } else {
                    UNROLL
                    for (int r = 0; r < 4; ++r) {
                        v[r] = sycl::bit_cast<uint32_t>(src[c * chunk + r * sgs + sgl]);
                    }
                    syclex::group_store(sg, v, reinterpret_cast<uint32_t*>(dst + c * chunk), props);
                }
            }
            done = nchunks * chunk;
        }
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        for (int i = done + lid; i < n; i += lsz) {
            dst[i] = src[i];
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
        const auto barrier = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        T reg[MAXNQ];

        group_copy<true>(it, b_in, w0, cb * nm3);
        barrier();
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
        barrier();
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
        barrier();
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
        barrier();
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
        barrier();
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
        barrier();
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
        barrier();
        group_copy<false>(it, w0, b_out, cb * nm3);
    }
};

template <typename T, bool block_io>
using sc_kernel = opt_kernel<T, sc_shape, block_io>;

// one launch over d.nelmt elements in the shape P (Shape::P for a
// fixed_shape).  With sc_shape: exe, the kernel built for P beforehand
// (bk::build_specialized<sc_kernel<T, block_io>, opt_id>), or nullptr to set
// the constant here (the first launch of a shape then JIT-compiles it)
template <typename T, typename Shape, bool block_io>
sycl::event launch(sycl::queue& q, const opt_params& P, const device_data<T>& d,
                   const sycl::kernel_bundle<sycl::bundle_state::executable>* exe = nullptr)
{
    return bk::launch<1>(q, opt_shape(P, block_io), d.nelmt, [&](sycl::handler& cgh) {
        if (exe != nullptr) {
            cgh.use_kernel_bundle(*exe);
        } else {
            Shape::set(cgh, P);
        }
        const sycl::local_accessor<T, 1> lm = bk::local_floats(cgh, opt_lmem(P));
        return opt_kernel<T, Shape, block_io>{d.basis, d.JxW_t, d.in, d.out, d.nelmt, lm};
    });
}

} // namespace bk1opt

#endif // BK1_SYCL_OPT_H
