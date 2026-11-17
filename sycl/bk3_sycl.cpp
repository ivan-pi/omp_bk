// bk3_sycl.cpp -- BK3 (Poisson operator) kernel variants in SYCL, one order
// per build (-DNQ=nq, nq = p + 2), every variant timed and checked against
// the double-precision CPU reference.  G stays in BK3's factor-major layout
// G(e, f, p, q, r); the steps are in include/bk3_gpu_steps.h.
//
//   scripts/build.sh bk3_sycl_<p>       (-> build/a770/bk3_sycl_<p>)
//   scripts/run.sh bk3_sycl_<p> [nelmt] [ntests]
//   BK_KERNEL="sf_b1cr_sg8 cu_b2_sg8"  run a subset (default: all; -name excludes)
//   BK_TEAMS=n                          cu: n work-groups stride over the batches
//
// Variants (b<batch>: elements per work-group):
//   cu_b<batch>    port of the CUDA kernel: BK1's interpolation steps around
//                  the derivative and flux steps, four nq^3 SLM boxes per
//                  element, B and D in SLM.  The CUDA source pairs rr with qt
//                  (Grr*qt + ... + Grt*qr); here the fluxes are BK3's.
//   sf_b<batch>    register-column sweeps through three SLM boxes, one
//                  work-item per (q, r) column; flags:
//                    c  carry the column in registers between steps 3, 5, 7
//                    p  box strides with the fewest bank conflicts
//                    r  B and D held in registers (pays off at SIMD8 only)
//   _sg8           sub-group size 8 (SIMD8: twice the registers per
//                  work-item; IGC picks SIMD16 otherwise)
// A variant whose batch is clamped to an earlier one's is the same kernel and
// is reported as such instead of being run again.
//
// Data: pseudo-random in and G (bk::fill_random_test_data), B(i,p) =
// cos(i * nq + p), D(n,p) = cos(n * nq + p).  GB/s counts in + out + G.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iterator>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"
#include "bk3_gpu_steps.h"

#ifndef NQ
#define NQ 6
#endif

namespace {

using local = sycl::local_accessor<float, 1>;

float* ptr(const local& a) { return a.get_multi_ptr<sycl::access::decorated::no>().get(); }

// ---------------------------------------------------------------------------
// cu: one accessor per SLM array, so the compiler knows that a store to a
// work box leaves B and D unchanged
// ---------------------------------------------------------------------------
template <int nq, int batch, int sg>
struct cu_kernel {
    static constexpr int nm = nq - 1;
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;
    static constexpr int nm3 = nm * nm * nm;
    // B, D and four work boxes in local memory; launched by bk::launch
    static constexpr bk::kernel_shape shape{batch, bk3gpu::cu_block(nq, batch), sg, 1,
                                            nm * nq + nq2 + 4 * batch * nq3};
    using interp = bk1gpu::gs_steps<float, nq>;
    using steps = bk3gpu::cu_steps<float, nq>;

    const float* basis;
    const float* dbasis;
    const float* G;
    const float* in;
    float* out;
    int nelmt;
    local l_basis;
    local l_dbasis;
    local l_wsp0;
    local l_wsp1;
    local l_rqr;
    local l_rqs;

    // the work-group size fixed at compile time (as the oneAPI GPU guide
    // advises) and a required sub-group size sg (0: IGC's choice)
    auto get(sycl::ext::oneapi::experimental::properties_tag) const
    {
        return bk::launch_properties<shape.sg, shape.block>();
    }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        float* s_basis = ptr(l_basis);
        float* s_dbasis = ptr(l_dbasis);
        float* w0 = ptr(l_wsp0);
        float* w1 = ptr(l_wsp1);
        float* rqr = ptr(l_rqr);
        float* rqs = ptr(l_rqs);
        float* rqt = w0;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        for (int t = lid; t < nm * nq; t += lsz) {
            s_basis[t] = basis[t];
        }
        for (int t = lid; t < nq2; t += lsz) {
            s_dbasis[t] = dbasis[t];
        }
        bar();
        const int nbatches = (nelmt + batch - 1) / batch;
        for (int eb = static_cast<int>(it.get_group(0)); eb < nbatches; eb += static_cast<int>(it.get_group_range(0))) {
            const int cb = std::min(batch, nelmt - eb * batch);
            const float* b_in = in + std::size_t(eb) * (batch * nm3);
            const float* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
            float* b_out = out + std::size_t(eb) * (batch * nm3);
            for (int x = lid; x < cb * nm3; x += lsz) {
                w0[x] = b_in[x];
            }
            bar();
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                interp::s2(t, s_basis, w0, w1);
            }
            bar();
            for (int t = lid; t < cb * nq * nm; t += lsz) {
                interp::s3(t, s_basis, w1, w0);
            }
            bar();
            for (int t = lid; t < cb * nq2; t += lsz) {
                interp::s4u(t, s_basis, w0, w1);
            }
            bar();
            for (int t = lid; t < cb * nq2; t += lsz) {
                steps::fluxes(t, s_dbasis, b_G, w1, rqr, rqs, rqt);
            }
            bar();
            for (int t = lid; t < cb * nq2; t += lsz) {
                steps::divergence(t, s_dbasis, rqr, rqs, rqt, w1);
            }
            bar();
            for (int t = lid; t < cb * nq2; t += lsz) {
                interp::s6(t, s_basis, w1, w0);
            }
            bar();
            for (int t = lid; t < cb * nm * nq; t += lsz) {
                interp::s7(t, s_basis, w0, w1);
            }
            bar();
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                interp::s8(t, s_basis, w1, w0);
            }
            bar();
            for (int x = lid; x < cb * nm3; x += lsz) {
                b_out[x] = w0[x];
            }
            bar();
        }
    }
};

// ---------------------------------------------------------------------------
// sf: one batch per work-group, one accessor per box
// ---------------------------------------------------------------------------
template <int nq, int batch, bool carry, bool padded, bool breg, int sg>
struct sf_kernel {
    using steps = bk3gpu::sf_steps<float, nq, padded>;
    static constexpr int nm = nq - 1;
    static constexpr int nq2 = nq * nq;
    static constexpr int nm3 = nm * nm * nm;
    // three boxes in local memory; launched by bk::launch
    static constexpr bk::kernel_shape shape{batch, bk3gpu::sf_block(nq, batch), sg, 1, 3 * batch * steps::ebox};
    static_assert(!carry || shape.block >= batch * nq2, "carry: one work-item per column");

    const float* gB;
    const float* gD;
    const float* G;
    const float* in;
    float* out;
    int nelmt;
    local lU;
    local lR;
    local lS;

    // the work-group size fixed at compile time, sg as in cu_kernel
    auto get(sycl::ext::oneapi::experimental::properties_tag) const
    {
        return bk::launch_properties<shape.sg, shape.block>();
    }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const int eb = static_cast<int>(it.get_group(0));
        const int cb = std::min(batch, nelmt - eb * batch);
        const float* b_in = in + std::size_t(eb) * (batch * nm3);
        const float* b_G = G + std::size_t(eb) * (batch * 6 * nq2 * nq);
        float* b_out = out + std::size_t(eb) * (batch * nm3);
        float* U = ptr(lU);
        float* R = ptr(lR);
        float* S = ptr(lS);
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        float Bv[nm * nq];
        float Dv[nq2];
        if constexpr (breg) {
            bk3gpu::load_basis<float, nq>(gB, gD, Bv, Dv);
        }
        const float* B = breg ? Bv : gB;
        const float* D = breg ? Dv : gD;
        float ucol[nq];   // carry: this work-item's (q, r) column

        for (int x = lid; x < cb * nm3; x += lsz) {
            steps::load(x, b_in, S);
        }
        bar();
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            steps::s1(t, B, S, R);
        }
        bar();
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            steps::s2(t, B, R, S);
        }
        bar();
        for (int t = lid; t < cb * nq2; t += lsz) {
            steps::template s3<carry>(t, B, S, U, ucol);
        }
        bar();
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            steps::s4(t, cb, D, U, R, S);
        }
        bar();
        for (int t = lid; t < cb * nq2; t += lsz) {
            steps::template s5<carry>(t, D, b_G, U, R, S, ucol);
        }
        bar();
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            steps::s6(t, cb, D, R, S);
        }
        bar();
        for (int t = lid; t < cb * nq2; t += lsz) {
            steps::template s7<carry>(t, B, U, R, S, ucol);
        }
        bar();
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            steps::s8(t, B, U, R);
        }
        bar();
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            steps::s9(t, B, R, S);
        }
        bar();
        for (int x = lid; x < cb * nm3; x += lsz) {
            steps::store(x, S, b_out);
        }
    }
};

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
struct device_data {
    float* basis;
    float* D;
    float* G;
    float* in;
    float* out;
    int nelmt;
};

using launch_fn = sycl::event (*)(sycl::queue&, const device_data&);

inline int g_grid = 0;   // BK_TEAMS

template <int batch, int sg>
sycl::event launch_cu(sycl::queue& q, const device_data& d)
{
    using K = cu_kernel<NQ, batch, sg>;
    constexpr int nq3 = NQ * NQ * NQ;
    const int nbatches = (d.nelmt + batch - 1) / batch;
    const int groups = (g_grid > 0 && g_grid < nbatches) ? g_grid : 0;   // BK_TEAMS: stride over the batches
    return bk::launch<K>(
        q, d.nelmt,
        [&](sycl::handler& cgh) {
            return K{d.basis,
                     d.D,
                     d.G,
                     d.in,
                     d.out,
                     d.nelmt,
                     bk::local_floats(cgh, (NQ - 1) * NQ),
                     bk::local_floats(cgh, NQ * NQ),
                     bk::local_floats(cgh, batch * nq3),
                     bk::local_floats(cgh, batch * nq3),
                     bk::local_floats(cgh, batch * nq3),
                     bk::local_floats(cgh, batch * nq3)};
        },
        groups);
}

template <int batch, int sg, bool carry, bool padded, bool breg>
sycl::event launch_sf(sycl::queue& q, const device_data& d)
{
    using K = sf_kernel<NQ, batch, carry, padded, breg, sg>;
    constexpr int ebox = bk3gpu::box_layout(NQ, padded).se;
    return bk::launch<K>(q, d.nelmt, [&](sycl::handler& cgh) {
        return K{d.basis,
                 d.D,
                 d.G,
                 d.in,
                 d.out,
                 d.nelmt,
                 bk::local_floats(cgh, batch * ebox),
                 bk::local_floats(cgh, batch * ebox),
                 bk::local_floats(cgh, batch * ebox)};
    });
}

struct variant {
    const char* name;
    launch_fn fn;
};

#define SG_SUFFIX_0 ""
#define SG_SUFFIX_8 "_sg8"
#define CU(B) {"cu_b" #B "_sg8", &launch_cu<bk3gpu::cu_batch(NQ, B), 8>}
// FLAGS spells the bools: c (carry), p (padded), r (breg)
#define SF(B, FLAGS, SG, C, P, R) \
    {"sf_b" #B FLAGS SG_SUFFIX_##SG, &launch_sf<bk3gpu::sf_batch(NQ, B, P), SG, C, P, R>}

const variant variants[] = {
    CU(1), CU(2), CU(4),
    SF(1, "", 0, false, false, false), SF(2, "", 0, false, false, false),
    SF(2, "c", 0, true, false, false), SF(4, "c", 0, true, false, false),
    SF(1, "r", 8, false, false, true), SF(2, "r", 8, false, false, true),
    SF(1, "cr", 8, true, false, true), SF(2, "cr", 8, true, false, true), SF(4, "cr", 8, true, false, true),
    SF(1, "cpr", 8, true, true, true), SF(2, "cpr", 8, true, true, true),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    g_grid = bk::env_int("BK_TEAMS", 0);

    // BK_PROFILE=0: no device times; BK_INORDER=1: an in-order queue
    sycl::queue q = bk::make_queue();
    std::printf("BK3 SYCL: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d, device %s\n", nq - 2, nq, nelmt,
                nelmt * nm * nm * nm, ntests, q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    std::vector<float> G(nelmt * 6 * nq * nq * nq);
    bk::fill_random_test_data(in, G);   // in in [-1, 1], G = 1.5 + [-1, 1]
    // BK_CHECK=0: benchmark only, no reference
    const bk::reference_check check(in.size(), [&](std::vector<double>& ref) {
        bk::reference_bk3(nq, nelmt, basis, D, G, in, ref);
    });

    device_data d{sycl::malloc_device<float>(basis.size(), q), sycl::malloc_device<float>(D.size(), q),
                  sycl::malloc_device<float>(G.size(), q), sycl::malloc_device<float>(in.size(), q),
                  sycl::malloc_device<float>(in.size(), q), static_cast<int>(nelmt)};
    q.copy(basis.data(), d.basis, basis.size());
    q.copy(D.data(), d.D, D.size());
    q.copy(G.data(), d.G, G.size());
    q.copy(in.data(), d.in, in.size());
    q.wait_and_throw();

    const double bytes = sizeof(float) * (2.0 * in.size() + G.size());
    std::vector<float> out(check.on() ? in.size() : 0);
    for (std::size_t v = 0; v < std::size(variants); ++v) {
        const variant& var = variants[v];
        if (!bk::selected(var.name)) {
            continue;
        }
        if (const variant* same = bk::same_kernel(variants, v)) {
            std::printf("%-12s = %s\n", var.name, same->name);
            continue;
        }
        q.memset(d.out, 0, in.size() * sizeof(float)).wait();
        bk::launch_times t{};
        if (!bk::time_variant(var.name, ntests, [&] { return var.fn(q, d); }, t)) {
            continue;
        }
        const std::string chk = check.text([&]() -> const std::vector<float>& {
            q.copy(d.out, out.data(), out.size()).wait();
            return out;
        });
        std::printf("%-12s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %s\n",
                    var.name, 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, chk.c_str());
    }
    sycl::free(d.basis, q);
    sycl::free(d.D, q);
    sycl::free(d.G, q);
    sycl::free(d.in, q);
    sycl::free(d.out, q);
    return 0;
}
