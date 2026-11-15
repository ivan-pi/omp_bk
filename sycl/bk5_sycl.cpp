// bk5_sycl.cpp -- BK5 (collocated Laplacian at quadrature points) in SYCL:
// the sum-factorised kernel sf of the OpenMP harness (openmp/offload/bk5_offload.cpp), a
// batch of elements per work-group with register-column sweeps through
// three local-memory arrays, for the symmetric half (G6) and the full (G9)
// geometric tensor.
//
//   scripts/build.sh bk5_sycl_<nq>      (-> build/a770/bk5_sycl_<nq>)
//   scripts/run.sh bk5_sycl_<nq> [nelmt] [ntests]
//   BK_KERNEL="sf6c_b1x1 sf9c_b1x1" ...         run a subset (default: all)
//
// Variant names sf<F>c_b<batch>x<w>[_sg16]:
//   F      6 (symmetric half: rr rs rt ss st tt) or 9 (full 3 x 3, row-major)
//   c      G in layout c: G(f, k, i, j) per element
//   batch  elements per work-group (capped so the three arrays fit 60 KB)
//   w      work-items per column of the batch (1 or 2), rounded to whole
//          sub-groups of 16 and capped at 1024
//   _sg16  sub-group size 16 required (otherwise the compiler's choice)
//   _2d    a 2D range (batches x work-group), as OpenMP offload launches
// BK_ALIGN=n allocates the device arrays with n-byte alignment.
// Steps per batch (U, R, S: nq^2 rows of pk floats per element, pk = nq or
// nq + 1 for even nq, so a k-row stride is odd):
//   copy   in -> U
//   A | B  item (e,j,k): R = D_i U along i;  item (e,i,k): S = D_j U along j
//   C      item (e,i,j): qt = D_k U along k; per point G applied to
//          (R, S, qt), rr -> R, rs -> S in place, rt kept in registers;
//          U(i,j,:) = D_k^T rt (only this item reads U(i,j,:) in C)
//   E | F  item (e,j,k): R = D_i^T R in place; item (e,i,k): S = D_j^T S
//   out    U + R + S -> out
// Barriers fence local memory only.  The step bodies are written out in the
// kernel (no helpers taking local pointers, which lose their address space).
//
// Data: pseudo-random in and G (bk::fill_random_test_data), D(i, n) =
// cos(i * nq + n), as in bk5_offload.cpp.  The check is against a
// double-precision CPU reference: the largest difference relative to the
// largest |reference|.  GB/s counts in + out + G (F floats per point).
// Host times include the launch; device times are from event profiling.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"

#ifndef NQ
#define NQ 7
#endif

namespace {

constexpr int round16(const int n) { return bk::round_up(n, 16); }

// (e, a, b) of a flat index t over an (*, na, nb) box, b fastest
struct idx3 {
    int e;
    int a;
    int b;
};

template <int na, int nb>
constexpr idx3 split3(const int t)
{
    return {t / (na * nb), t % (na * nb) / nb, t % nb};
}

// fluxes from the derivatives at a point: (qr, qs, qt) -> (rr, rs, rt)
template <typename T, int F>
inline void apply_g(const T (&g)[F], const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    if constexpr (F == 6) {
        rr = g[0] * qr + g[1] * qs + g[2] * qt;
        rs = g[1] * qr + g[3] * qs + g[4] * qt;
        rt = g[2] * qr + g[4] * qs + g[5] * qt;
    } else {
        rr = g[0] * qr + g[1] * qs + g[2] * qt;
        rs = g[3] * qr + g[4] * qs + g[5] * qt;
        rt = g[6] * qr + g[7] * qs + g[8] * qt;
    }
}

constexpr int k_stride(const int nq) { return (nq % 2 == 0) ? nq + 1 : nq; }

// elements per work-group: at most `want`, three arrays within 60 KB
constexpr int sf_batch(const int nq, const int want)
{
    const int by_slm = 61440 / (3 * nq * nq * k_stride(nq) * 4);
    const int b = (want < by_slm) ? want : by_slm;
    return (b > 1) ? b : 1;
}

// work-items per group: `width` per column of the batch, whole sub-groups,
// at most 1024
constexpr int sf_block(const int nq, const int batch, const int width)
{
    const int w = round16(width * batch * nq * nq);
    return (w < 1024) ? w : 1024;
}

template <typename T, int nq, int F, int batch>
struct sf_kernel {
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;
    static constexpr int pk = k_stride(nq);
    static constexpr int ebox = nq2 * pk;
    static constexpr int lmem = 3 * batch * ebox;
    static_assert(lmem * int(sizeof(T)) <= 61440, "three local arrays within 60 KB");

    const T* D;
    const T* G;
    const T* in;
    T* out;
    int nelmt;
    sycl::local_accessor<T, 1> lm;

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { run(it); }

    // the work-group is the last dimension of the range, the batch the first
    template <int dims>
    void run(sycl::nd_item<dims> it) const
    {
        const int lid = static_cast<int>(it.get_local_id(dims - 1));
        const int lsz = static_cast<int>(it.get_local_range(dims - 1));
        const int eb = static_cast<int>(it.get_group(0));
        const int remaining = nelmt - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nq3);
        const T* b_G = G + std::size_t(eb) * (batch * F * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nq3);
        T* U = lm.template get_multi_ptr<sycl::access::decorated::no>().get();
        T* R = U + batch * ebox;
        T* S = R + batch * ebox;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };

        for (int x = lid; x < cb * nq3; x += lsz) {
            U[x / nq * pk + x % nq] = b_in[x];
        }
        bar();
        // A | B
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            T col[nq];
            if (t < cb * nq2) {
                const auto [e, j, k] = split3<nq, nq>(t);
                const T* eu = U + e * ebox + j * pk + k;
                T* er = R + e * ebox + j * pk + k;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = eu[n * nq * pk];
                }
                #pragma unroll
                for (int i = 0; i < nq; ++i) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[i * nq + n] * col[n];
                    }
                    er[i * nq * pk] = s;
                }
            } else {
                const auto [e, i, k] = split3<nq, nq>(t - cb * nq2);
                const T* eu = U + e * ebox + i * nq * pk + k;
                T* es = S + e * ebox + i * nq * pk + k;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = eu[n * pk];
                }
                #pragma unroll
                for (int j = 0; j < nq; ++j) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[j * nq + n] * col[n];
                    }
                    es[j * pk] = s;
                }
            }
        }
        bar();
        // C
        for (int t = lid; t < cb * nq2; t += lsz) {
            const int e = t / nq2;
            const int c = t % nq2;
            const int base = e * ebox + c * pk;
            const T* eG = b_G + e * (F * nq3) + c;
            T col[nq];
            T rt[nq];
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                col[n] = U[base + n];
            }
            #pragma unroll
            for (int k = 0; k < nq; ++k) {
                T qt = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qt += D[k * nq + n] * col[n];
                }
                T g[F];
                #pragma unroll
                for (int f = 0; f < F; ++f) {
                    g[f] = eG[(f * nq + k) * nq2];
                }
                T rr;
                T rs;
                apply_g<T, F>(g, R[base + k], S[base + k], qt, rr, rs, rt[k]);
                R[base + k] = rr;
                S[base + k] = rs;
            }
            #pragma unroll
            for (int k = 0; k < nq; ++k) {
                T s = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    s += D[n * nq + k] * rt[n];
                }
                U[base + k] = s;
            }
        }
        bar();
        // E | F
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            T col[nq];
            if (t < cb * nq2) {
                const auto [e, j, k] = split3<nq, nq>(t);
                T* er = R + e * ebox + j * pk + k;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = er[n * nq * pk];
                }
                #pragma unroll
                for (int i = 0; i < nq; ++i) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[n * nq + i] * col[n];
                    }
                    er[i * nq * pk] = s;
                }
            } else {
                const auto [e, i, k] = split3<nq, nq>(t - cb * nq2);
                T* es = S + e * ebox + i * nq * pk + k;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = es[n * pk];
                }
                #pragma unroll
                for (int j = 0; j < nq; ++j) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[n * nq + j] * col[n];
                    }
                    es[j * pk] = s;
                }
            }
        }
        bar();
        for (int x = lid; x < cb * nq3; x += lsz) {
            const int y = x / nq * pk + x % nq;
            b_out[x] = U[y] + R[y] + S[y];
        }
    }
};

// a 2D range (batches x work-group), the shape OpenMP offload launches
template <typename K>
struct as2d {
    K k;
    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<2> it) const { k.run(it); }
};

template <typename K>
struct sg16 {
    K k;
    [[sycl::reqd_sub_group_size(16), intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k(it); }
};

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
struct device_data {
    float* D;
    float* G[2];   // G6 and G9 in layout c
    float* in;
    float* out;
    int nelmt;
};

using launch_fn = sycl::event (*)(sycl::queue&, const device_data&);

template <int F, int batch, int block, bool sub16, bool two_d = false>
sycl::event launch_sf(sycl::queue& q, const device_data& d)
{
    using K = sf_kernel<float, NQ, F, batch>;
    static_assert(block <= 1024 && block % 16 == 0, "whole sub-groups, at most 1024 work-items");
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * block, std::size_t(block)};
    const float* G = d.G[F == 6 ? 0 : 1];
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>(K::lmem), cgh);
        const K k{d.D, G, d.in, d.out, d.nelmt, lm};
        if constexpr (two_d) {
            const sycl::nd_range<2> range2{{std::size_t(ngroups), std::size_t(block)}, {1, std::size_t(block)}};
            cgh.parallel_for(range2, as2d<K>{k});
        } else if constexpr (sub16) {
            cgh.parallel_for(range, sg16<K>{k});
        } else {
            cgh.parallel_for(range, k);
        }
    });
}

struct variant {
    const char* name;
    launch_fn fn;
    int F;
};

#define SF(F, B, W, SG, SUFFIX) \
    {"sf" #F "c_b" #B "x" #W SUFFIX, &launch_sf<F, sf_batch(NQ, B), sf_block(NQ, sf_batch(NQ, B), W), SG>, F}
#define SF2(F, B, W) \
    {"sf" #F "c_b" #B "x" #W "_2d", &launch_sf<F, sf_batch(NQ, B), sf_block(NQ, sf_batch(NQ, B), W), false, true>, F}

const variant variants[] = {
    SF(6, 1, 1, false, ""), SF(6, 2, 1, false, ""), SF(6, 4, 1, false, ""), SF(6, 8, 1, false, ""),
    SF(6, 16, 1, false, ""), SF(6, 32, 1, false, ""),
    SF(6, 1, 2, false, ""), SF(6, 2, 2, false, ""), SF(6, 4, 2, false, ""), SF(6, 8, 2, false, ""),
    SF(6, 16, 2, false, ""), SF(6, 32, 2, false, ""),
    SF(6, 1, 1, true, "_sg16"), SF(6, 1, 2, true, "_sg16"), SF(6, 4, 2, true, "_sg16"),
    SF(9, 1, 1, false, ""), SF(9, 2, 1, false, ""), SF(9, 4, 1, false, ""), SF(9, 8, 1, false, ""),
    SF(9, 16, 1, false, ""), SF(9, 32, 1, false, ""),
    SF(9, 1, 2, false, ""), SF(9, 2, 2, false, ""), SF(9, 4, 2, false, ""), SF(9, 8, 2, false, ""),
    SF(9, 16, 2, false, ""), SF(9, 32, 2, false, ""),
    SF(9, 1, 1, true, "_sg16"), SF(9, 1, 2, true, "_sg16"),
    SF2(6, 1, 1), SF2(6, 8, 2), SF2(9, 1, 1), SF2(9, 4, 2),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nq3 = nq * nq * nq;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;

    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}}};
    std::printf("BK5 SYCL: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d, device %s\n", nq - 1, nq, nelmt,
                nelmt * nq3, ntests, q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nq3);
    std::vector<float> G6i(nelmt * 6 * nq3);
    bk::fill_random_test_data(in, G6i);   // in in [-1, 1], G6i = 1.5 + [-1, 1]
    std::vector<double> ref(in.size());
    bk::reference_bk5(nq, nelmt, D, G6i, in, ref);
    const double big = bk::max_abs(ref);

    // BK_ALIGN=n: device arrays from aligned_alloc_device with n-byte
    // alignment (default: malloc_device)
    const std::size_t align = std::size_t(bk::env_int("BK_ALIGN", 0));
    const auto dev_alloc = [&](const std::size_t n) {
        return (align > 0) ? sycl::aligned_alloc_device<float>(align, n, q) : sycl::malloc_device<float>(n, q);
    };

    // a G is built and copied only if a selected variant uses it
    bool need[2] = {false, false};
    for (const variant& v : variants) {
        if (bk::selected(v.name)) {
            need[v.F == 6 ? 0 : 1] = true;
        }
    }
    device_data d{dev_alloc(D.size()), {nullptr, nullptr}, dev_alloc(in.size()), dev_alloc(in.size()),
                  static_cast<int>(nelmt)};
    q.copy(D.data(), d.D, D.size());
    q.copy(in.data(), d.in, in.size());
    for (int g = 0; g < 2; ++g) {
        if (need[g]) {
            const std::vector<float> Gc = bk::make_bk5_G(G6i, nelmt, nq, g == 0 ? 6 : 9, 2);
            d.G[g] = dev_alloc(Gc.size());
            q.copy(Gc.data(), d.G[g], Gc.size()).wait_and_throw();
        }
    }
    q.wait_and_throw();
    G6i = std::vector<float>();

    std::vector<float> out(in.size());
    for (const variant& v : variants) {
        if (!bk::selected(v.name)) {
            continue;
        }
        q.memset(d.out, 0, out.size() * sizeof(float)).wait();
        const bk::launch_times t = bk::time_launches(ntests, [&] { return v.fn(q, d); });
        q.copy(d.out, out.data(), out.size()).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        const double bytes = sizeof(float) * double(nelmt) * nq3 * (2.0 + v.F);
        std::printf("%-15s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s\n",
                    v.name, 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, relerr, bk::fail_mark(relerr));
    }
    sycl::free(d.D, q);
    for (float* g : d.G) {
        if (g != nullptr) {
            sycl::free(g, q);
        }
    }
    sycl::free(d.in, q);
    sycl::free(d.out, q);
    return 0;
}
