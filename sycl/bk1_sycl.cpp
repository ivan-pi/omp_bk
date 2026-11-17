// bk1_sycl.cpp -- BK1 (mass operator) in SYCL, variants of the seven-step
// structure of the CUDA/Kokkos kernels (include/bk1_gpu_steps.h): a batch of
// elements per work-group, contiguous copies of in and out, the contractions
// through local memory, the basis read from global memory and JxW stored
// transposed per element, JxW_t(e, r, q, p) = JxW(e, p, q, r).
//
//   scripts/build.sh bk1_sycl_<p>       (-> build/a770/bk1_sycl_<p>, nq = p + 2)
//   scripts/build.sh bk1_sycl_<p>_jit   (JIT, for spec)
//   scripts/run.sh bk1_sycl_<p> [nelmt] [ntests]
//   BK_KERNEL="C_l_sg0_b1 spec_b1" ...   run a subset (default: all)
//   BK_PROFILE=0 (no device times), BK_INORDER=1 (in-order queue)
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
// inl*: the opt kernel of sycl/bk1_sycl_opt.h in the shapes of
//   bk1opt::opt_variants (inl_b<B>: dense SLM, P padded, B sub-group block
//   copies, F_f<T> a group of T work-items filled with elements).
// spec_b1/b4: inl_b1/b4 with the shape from a specialization constant; use
//   the bk1_sycl_<p>_jit build for it to be folded (ahead of time it is a
//   run-time value, 2.5-5x slower).  bk1_sycl_sc runs every shape and order
//   this way from one binary.
//
// Data and check: sycl/bk1_sycl_harness.h.

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"
#include "bk1_gpu_steps.h"
#include "bk1_sycl_harness.h"

#ifndef NQ
#define NQ 6
#endif

namespace {

using bk1gpu::gs_steps;


// ---------------------------------------------------------------------------
// the seven-step kernel, compile-time nq
// ---------------------------------------------------------------------------
// sg: the required sub-group size (0: IGC's choice)
template <typename T, int nq, int batch, bool copy, bool local_fence, int sg>
struct seven_step {
    static constexpr int nm = nq - 1;
    static constexpr int nq3 = nq * nq * nq;
    static constexpr int nm3 = nm * nm * nm;
    static constexpr int block = bk1gpu::gs_block(nq, batch);
    static constexpr int lmem = batch * (nq3 + nq3);

    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    auto get(sycl::ext::oneapi::experimental::properties_tag) const { return bk::launch_properties<sg>(); }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { run(it); }

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

// ---------------------------------------------------------------------------
// one element per work-item (the reference SumFactorization's structure, as
// the OpenMP `teams loop` kernel): private work arrays, no local memory and
// no barriers; the lanes of a sub-group run across elements.  Fast while the
// 2 * nq^3 private floats stay in registers (p = 1, 2); from p = 4 they spill.
// JxW is the transposed per-element layout the harness passes to all
// kernels, JxW_t(e, r, q, p).
// ---------------------------------------------------------------------------
// wg: work-items per group of an nd_range launch (0: a plain range, wg
// left to the runtime); sg: the required sub-group size (0: IGC's choice)
template <typename T, int nq, bool vec_io, int wg, int sg>
struct one_element {
    const T* basis;
    const T* JxW_t;
    const T* in;
    T* out;
    int nelmt;

    auto get(sycl::ext::oneapi::experimental::properties_tag) const { return bk::launch_properties<sg>(); }

    [[intel::kernel_args_restrict]] void operator()(sycl::item<1> it) const { run(static_cast<int>(it.get_id(0))); }
    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        run(static_cast<int>(it.get_global_id(0)));
    }

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

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
using bk1sycl::device_data;

using launch_fn = sycl::event (*)(sycl::queue&, const device_data&);

template <int nq, int batch, bool copy, bool local_fence, int sg>
sycl::event launch(sycl::queue& q, const device_data& d)
{
    using K = seven_step<float, nq, batch, copy, local_fence, sg>;
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * K::block, std::size_t(K::block)};
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>(K::lmem), cgh);
        cgh.parallel_for(range, K{d.basis, d.JxW_t, d.in, d.out, d.nelmt, lm});
    });
}

// the opt kernel (sycl/bk1_sycl_opt.h) in shape bk1opt::opt_variants[i] for
// nq = NQ, fixed at compile time; sc: the same shape from the
// specialization constant (folded by a JIT build, a run-time value ahead of
// time)
template <int i, bool sc = false>
sycl::event launch_opt(sycl::queue& q, const device_data& d)
{
    constexpr bk1opt::opt_variant v = bk1opt::opt_variants[i];
    constexpr bk1opt::opt_params P = bk1opt::variant_params(NQ, v);
    static_assert(bk1opt::opt_valid(P, v.block_io), "one work-item per point of the widest step, whole sub-groups");
    using S = std::conditional_t<sc, bk1opt::sc_shape, bk1opt::fixed_shape<NQ, P.batch, P.block, v.slm>>;
    return bk1opt::launch<float, S, v.block_io>(q, P, d);
}

template <int wg, int sg = 0, bool vec_io = false>
sycl::event launch_elem(sycl::queue& q, const device_data& d)
{
    const one_element<float, NQ, vec_io, wg, sg> k{d.basis, d.JxW_t, d.in, d.out, d.nelmt};
    return q.submit([&](sycl::handler& cgh) {
        if constexpr (wg == 0) {
            cgh.parallel_for(sycl::range<1>(std::size_t(d.nelmt)), k);
        } else {
            const std::size_t global = (std::size_t(d.nelmt) + wg - 1) / wg * wg;
            cgh.parallel_for(sycl::nd_range<1>(global, wg), k);
        }
    });
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
    {"spec_b1", &launch_opt<0, true>},
    {"spec_b4", &launch_opt<1, true>},
};

// then the opt kernel in every shape of bk1opt::opt_variants (inl*)
template <std::size_t... I>
void add_opt(std::vector<bk1sycl::variant>& run, std::index_sequence<I...>)
{
    (run.push_back({bk1opt::opt_variants[I].aot, &launch_opt<int(I)>, ""}), ...);
}

#if !defined(BK_NO_BASELINES) || NQ <= 4
const variant elem_variants[] = {
    // one element per work-item: the fastest kernels at p = 1 and close
    // behind at p = 2 on the A770 (results/a770/variant_survey_a770.csv), but
    // 4x to 100x slower than the batched kernels from p = 3, expensive to
    // compile there (register spills) and long enough to trip a small GPU's
    // hang check (UHD 750, p = 6): -DBK_NO_BASELINES leaves them out for
    // nq > 4 only
    {"elem", &launch_elem<0>},
    {"elem_wg128", &launch_elem<128>},
    {"elem_sg16", &launch_elem<128, 16>},
    {"elem_sg32", &launch_elem<128, 32>},
    {"elem_v4", &launch_elem<128, 0, true>},
};
#endif

} // namespace

int main(int argc, char** argv)
{
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : bk::default_nelmt;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 20;
    std::vector<bk1sycl::variant> run;
    for (const variant& v : variants) {
        run.push_back({v.name, v.fn, ""});
    }
    add_opt(run, std::make_index_sequence<std::size(bk1opt::opt_variants)>{});
#if !defined(BK_NO_BASELINES) || NQ <= 4
    for (const variant& v : elem_variants) {
        run.push_back({v.name, v.fn, ""});
    }
#endif
    sycl::queue q = bk::make_queue();
    bk1sycl::run(q, "BK1 SYCL", NQ, nelmt, ntests, run);
    return 0;
}
