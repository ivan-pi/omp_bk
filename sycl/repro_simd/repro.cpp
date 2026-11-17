// repro.cpp -- reproducer: on an Arc A770 (DG2), IGC's automatic SIMD width
// for this SYCL kernel is 1.2-1.5x slower than a narrower one, and an
// unrelated source change (wrapping the kernel functor in another functor)
// flips the choice.
//
// The kernel is BK5's sum-factorized Laplacian, variant sf9c_b1x1 of
// sycl/bk5_sycl.cpp in https://github.com/ivan-pi/omp_bk (the body copied
// verbatim): one element per work-group, one work-item per column of the
// nq x nq x nq points, three work arrays in local memory, and a full 3 x 3
// geometric tensor (9 values per point) read from global memory.  The same
// body is compiled into several kernels that differ only in how they are
// declared or launched (the cases below); all run on the same data, must give
// identical results, and are timed by event profiling (best of ntests
// launches after a warm-up).
//
//   ./build.sh                 -> repro_3 and repro_9 (nq = 3 and 9), and
//                                 IGC's assembly in asm_3/, asm_9/
//   ./repro_<nq> [nelmt] [ntests]
//
// No other code from the repository is needed.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#ifndef NQ
#define NQ 9
#endif

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

constexpr int nq = NQ;
constexpr int nq2 = nq * nq;
constexpr int nq3 = nq2 * nq;
constexpr int F = 9;                                   // tensor factors per point
constexpr int batch = 1;                               // elements per work-group
constexpr int pk = (nq % 2 == 0) ? nq + 1 : nq;        // odd k-row stride in local memory
constexpr int ebox = nq2 * pk;                         // floats per local array and element
constexpr int block = (batch * nq2 + 15) / 16 * 16;    // one work-item per column

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
inline void apply_g(const float (&g)[F], const float qr, const float qs, const float qt, float& rr, float& rs,
                    float& rt)
{
    rr = g[0] * qr + g[1] * qs + g[2] * qt;
    rs = g[3] * qr + g[4] * qs + g[5] * qt;
    rt = g[6] * qr + g[7] * qs + g[8] * qt;
}

// the kernel body (sf_kernel::run of sycl/bk5_sycl.cpp, verbatim), written
// out in every kernel's operator(); Dims is the range's dimension
#define KERNEL_BODY \
    const int lid = static_cast<int>(it.get_local_id(Dims - 1));                                \
    const int lsz = static_cast<int>(it.get_local_range(Dims - 1));                             \
    const int eb = static_cast<int>(it.get_group(0));                                    \
    const int remaining = nelmt - eb * batch;                                            \
    const int cb = (batch < remaining) ? batch : remaining;                              \
    const float* b_in = in + std::size_t(eb) * (batch * nq3);                            \
    const float* b_G = G + std::size_t(eb) * (batch * F * nq3);                          \
    float* b_out = out + std::size_t(eb) * (batch * nq3);                                \
    float* U = lm.get_multi_ptr<sycl::access::decorated::no>().get();                    \
    float* R = U + batch * ebox;                                                         \
    float* S = R + batch * ebox;                                                         \
    const auto barrier = [&]() { it.barrier(sycl::access::fence_space::local_space); };  \
                                                                                         \
    for (int x = lid; x < cb * nq3; x += lsz) {                                          \
        U[x / nq * pk + x % nq] = b_in[x];                                               \
    }                                                                                    \
    barrier();                                                                           \
    /* A | B */                                                                          \
    for (int t = lid; t < 2 * cb * nq2; t += lsz) {                                      \
        float col[nq];                                                                   \
        if (t < cb * nq2) {                                                              \
            const auto [e, j, k] = split3<nq, nq>(t);                                    \
            const float* eu = U + e * ebox + j * pk + k;                                 \
            float* er = R + e * ebox + j * pk + k;                                       \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                col[n] = eu[n * nq * pk];                                                \
            }                                                                            \
            _Pragma("unroll")                                                            \
            for (int i = 0; i < nq; ++i) {                                               \
                float s = 0;                                                             \
                _Pragma("unroll")                                                        \
                for (int n = 0; n < nq; ++n) {                                           \
                    s += D[i * nq + n] * col[n];                                         \
                }                                                                        \
                er[i * nq * pk] = s;                                                     \
            }                                                                            \
        } else {                                                                         \
            const auto [e, i, k] = split3<nq, nq>(t - cb * nq2);                         \
            const float* eu = U + e * ebox + i * nq * pk + k;                            \
            float* es = S + e * ebox + i * nq * pk + k;                                  \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                col[n] = eu[n * pk];                                                     \
            }                                                                            \
            _Pragma("unroll")                                                            \
            for (int j = 0; j < nq; ++j) {                                               \
                float s = 0;                                                             \
                _Pragma("unroll")                                                        \
                for (int n = 0; n < nq; ++n) {                                           \
                    s += D[j * nq + n] * col[n];                                         \
                }                                                                        \
                es[j * pk] = s;                                                          \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
    barrier();                                                                           \
    /* C */                                                                              \
    for (int t = lid; t < cb * nq2; t += lsz) {                                          \
        const int e = t / nq2;                                                           \
        const int c = t % nq2;                                                           \
        const int base = e * ebox + c * pk;                                              \
        const float* eG = b_G + e * (F * nq3) + c;                                       \
        float col[nq];                                                                   \
        float rt[nq];                                                                    \
        _Pragma("unroll")                                                                \
        for (int n = 0; n < nq; ++n) {                                                   \
            col[n] = U[base + n];                                                        \
        }                                                                                \
        _Pragma("unroll")                                                                \
        for (int k = 0; k < nq; ++k) {                                                   \
            float qt = 0;                                                                \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                qt += D[k * nq + n] * col[n];                                            \
            }                                                                            \
            float g[F];                                                                  \
            _Pragma("unroll")                                                            \
            for (int f = 0; f < F; ++f) {                                                \
                g[f] = eG[(f * nq + k) * nq2];                                           \
            }                                                                            \
            float rr;                                                                    \
            float rs;                                                                    \
            apply_g(g, R[base + k], S[base + k], qt, rr, rs, rt[k]);           \
            R[base + k] = rr;                                                            \
            S[base + k] = rs;                                                            \
        }                                                                                \
        _Pragma("unroll")                                                                \
        for (int k = 0; k < nq; ++k) {                                                   \
            float s = 0;                                                                 \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                s += D[n * nq + k] * rt[n];                                              \
            }                                                                            \
            U[base + k] = s;                                                             \
        }                                                                                \
    }                                                                                    \
    barrier();                                                                           \
    /* E | F */                                                                          \
    for (int t = lid; t < 2 * cb * nq2; t += lsz) {                                      \
        float col[nq];                                                                   \
        if (t < cb * nq2) {                                                              \
            const auto [e, j, k] = split3<nq, nq>(t);                                    \
            float* er = R + e * ebox + j * pk + k;                                       \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                col[n] = er[n * nq * pk];                                                \
            }                                                                            \
            _Pragma("unroll")                                                            \
            for (int i = 0; i < nq; ++i) {                                               \
                float s = 0;                                                             \
                _Pragma("unroll")                                                        \
                for (int n = 0; n < nq; ++n) {                                           \
                    s += D[n * nq + i] * col[n];                                         \
                }                                                                        \
                er[i * nq * pk] = s;                                                     \
            }                                                                            \
        } else {                                                                         \
            const auto [e, i, k] = split3<nq, nq>(t - cb * nq2);                         \
            float* es = S + e * ebox + i * nq * pk + k;                                  \
            _Pragma("unroll")                                                            \
            for (int n = 0; n < nq; ++n) {                                               \
                col[n] = es[n * pk];                                                     \
            }                                                                            \
            _Pragma("unroll")                                                            \
            for (int j = 0; j < nq; ++j) {                                               \
                float s = 0;                                                             \
                _Pragma("unroll")                                                        \
                for (int n = 0; n < nq; ++n) {                                           \
                    s += D[n * nq + j] * col[n];                                         \
                }                                                                        \
                es[j * pk] = s;                                                          \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
    barrier();                                                                           \
    for (int x = lid; x < cb * nq3; x += lsz) {                                          \
        const int y = x / nq * pk + x % nq;                                              \
        b_out[x] = U[y] + R[y] + S[y];                                                   \
    }                                                                                    \

// The kernels.  Restrict: [[intel::kernel_args_restrict]] (it makes no
// difference here); SG: a required sub-group size, i.e. the SIMD width
// (0: IGC's choice); Dims: a 1D range or a 2D one (groups x work-group).
template <bool Restrict, int SG, int Dims>
struct kernel;

template <int SG, int Dims>
struct kernel<true, SG, Dims> {
    const float* D;
    const float* G;
    const float* in;
    float* out;
    int nelmt;
    sycl::local_accessor<float, 1> lm;

    auto get(syclex::properties_tag) const
    {
        if constexpr (SG > 0) {
            return syclex::properties{syclex::sub_group_size<SG>};
        } else {
            return syclex::properties{};
        }
    }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<Dims> it) const
    {
        KERNEL_BODY
    }
};

template <int SG, int Dims>
struct kernel<false, SG, Dims> {
    const float* D;
    const float* G;
    const float* in;
    float* out;
    int nelmt;
    sycl::local_accessor<float, 1> lm;

    auto get(syclex::properties_tag) const
    {
        if constexpr (SG > 0) {
            return syclex::properties{syclex::sub_group_size<SG>};
        } else {
            return syclex::properties{};
        }
    }

    void operator()(sycl::nd_item<Dims> it) const
    {
        KERNEL_BODY
    }
};

// the same kernel as a member of a wrapper functor (2D range, IGC's SIMD
// width): the form that happened to be fast
struct wrapped {
    kernel<false, 0, 2> k;
    void operator()(sycl::nd_item<2> it) const { k(it); }
};

struct data {
    float* D;
    float* G;
    float* in;
    float* out;
    int nelmt;
};

// best device time of ntests launches (after a warm-up), in ms; make builds
// the kernel functor from the local accessor (make)
template <int Dims, typename Make>
double best_ms(sycl::queue& q, const data& d, const int ntests, Make make)
{
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t <= ntests; ++t) {
        sycl::event ev = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lm(sycl::range<1>(3 * batch * ebox), cgh);
            if constexpr (Dims == 1) {
                cgh.parallel_for(sycl::nd_range<1>(std::size_t(d.nelmt) * block, block), make(lm));
            } else {
                cgh.parallel_for(sycl::nd_range<2>({std::size_t(d.nelmt), block}, {1, block}), make(lm));
            }
        });
        ev.wait_and_throw();
        if (t > 0) {
            const double ns = double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                     ev.get_profiling_info<sycl::info::event_profiling::command_start>());
            best = std::min(best, 1e-6 * ns);
        }
    }
    return best;
}

// one case: its time, and whether its result equals ref (the first case's)
template <int Dims, typename Make>
bool report(sycl::queue& q, const data& d, const int ntests, const char* what, Make make, std::vector<float>& ref)
{
    const double ms = best_ms<Dims>(q, d, ntests, make);
    std::vector<float> res(std::size_t(d.nelmt) * nq3);
    q.copy(d.out, res.data(), res.size()).wait();
    if (ref.empty()) {
        ref = res;
    }
    const bool same = res == ref;
    std::printf("  %-44s %8.3f ms  %6.2f GDoF/s%s\n", what, ms, 1e-6 * double(d.nelmt) * nq3 / ms,
                same ? "" : "  result DIFFERS");
    return same;
}

} // namespace

int main(int argc, char** argv)
{
    const int nelmt = (argc > 1) ? std::atoi(argv[1]) : static_cast<int>(2e7 / nq3);
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 20;
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}}};
    std::printf("nq = %d, nelmt = %d (%d DoFs), work-group %d, device %s\n", nq, nelmt, nelmt * nq3, block,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<float> D(nq2);
    std::vector<float> in(std::size_t(nelmt) * nq3);
    std::vector<float> G(std::size_t(nelmt) * F * nq3);
    for (int i = 0; i < nq2; ++i) {
        D[i] = static_cast<float>(std::cos(double(i)));
    }
    unsigned seed = 1;
    const auto next = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / float(1u << 24) * 2.0f - 1.0f;
    };
    for (float& v : in) {
        v = next();
    }
    for (float& v : G) {
        v = 1.5f + next();
    }

    const data d{sycl::malloc_device<float>(D.size(), q), sycl::malloc_device<float>(G.size(), q),
                 sycl::malloc_device<float>(in.size(), q), sycl::malloc_device<float>(in.size(), q), nelmt};
    q.copy(D.data(), d.D, D.size());
    q.copy(G.data(), d.G, G.size());
    q.copy(in.data(), d.in, in.size()).wait();

    using acc = sycl::local_accessor<float, 1>;
    std::vector<float> ref;
    bool same = true;
    std::printf("IGC's SIMD width:\n");
    same = report<1>(q, d, ntests, "1D range, kernel_args_restrict",
                     [&](acc lm) { return kernel<true, 0, 1>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;
    same = report<1>(q, d, ntests, "1D range",
                     [&](acc lm) { return kernel<false, 0, 1>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;
    same = report<2>(q, d, ntests, "2D range",
                     [&](acc lm) { return kernel<false, 0, 2>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;
    same = report<2>(q, d, ntests, "2D range, in a wrapper functor",
                     [&](acc lm) { return wrapped{{d.D, d.G, d.in, d.out, nelmt, lm}}; }, ref) && same;
    std::printf("required sub-group size (SIMD width), 1D range:\n");
    same = report<1>(q, d, ntests, "8",
                     [&](acc lm) { return kernel<false, 8, 1>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;
    same = report<1>(q, d, ntests, "16",
                     [&](acc lm) { return kernel<false, 16, 1>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;
    same = report<1>(q, d, ntests, "32",
                     [&](acc lm) { return kernel<false, 32, 1>{d.D, d.G, d.in, d.out, nelmt, lm}; }, ref) && same;

    sycl::free(d.D, q);
    sycl::free(d.G, q);
    sycl::free(d.in, q);
    sycl::free(d.out, q);
    return same ? 0 : 1;
}
