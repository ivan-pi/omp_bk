// bk_tinytc.cpp -- BK1, BK3 and BK5 kernels written for the Intel Tiny Tensor
// Compiler (tinytc), run on the data and checked against the
// double-precision references of this repository's harness
// (include/bk_harness.h).
//
// TT_LAYOUT=ec (default): tinytc/ir/bk1_ec.ir, bk3_ec.ir and bk5_ec.ir, the
// kernels for our element-contiguous arrays, which they take unchanged.
// TT_LAYOUT=blk: the original kernels, tinytc/ir/bk1.ir and bk5_blk.ir
// (bk5.ir with its global-memory race fixed), and bk3_blk.ir, on our arrays
// converted to their blocked layout (below) and the result converted back.
// TT_LAYOUT=orig: as blk with the unmodified bk5.ir.
// TT_LAYOUT=blk_ch (BK3): bk3_blk_ch.ir, blk with the r direction chained in
// registers and 3 B nm nq^2 floats of local memory instead of 3 B nq^3.
//
//   scripts/build.sh bk_tinytc              (-> build/a770/bk_tinytc)
//   build/a770/bk_tinytc <1|3|5> <p> [nelmt] [ntests]   (BK1, BK3: nq = p + 2, BK5: nq = p + 1)
//   env: TT_LAYOUT (ec, blk or blk_ch), TT_IR_DIR (default: tinytc/ir of the source tree),
//        TT_QPAD (row stride of the basis, default: the next power of two >= nq),
//        TT_FEATURES (tinytc core feature flags, e.g. 0 for none),
//        TT_SPV (write the generated SPIR-V to this file),
//        TT_B (elements per block, default 16), TT_WGS (work-group size,
//        default: the table of the original driver), TT_SG (BK3: sub-group
//        size = rows of a cooperative-matrix tile, 8 or 16 (default); TT_B
//        a multiple of it), TT_SPILL (BK3: 1 keeps S in global memory,
//        bk3_*_gs.ir, so local memory holds 2 B nq^3 floats instead of 3;
//        default: only where 3 B nq^3 floats exceed 64 KiB)
//
// Conventions.  Ours: element-contiguous in/out (BK1 in(e,i,j,k) of nm^3
// modes, BK5 in(e,i,j,k) of nq^3 points), BK1 JxW(e,p,q,r), BK5 G6
// interleaved G(e,i,j,f,k), BK1 basis(i,p) = cos(i nq + p), BK5 D(i,n) =
// cos(i nq + n), BK3 as BK1 plus D(n,p) = cos(n nq + p) and G6 factor-major
// G(e,f,p,q,r).  tinytc's kernels take blocks of B elements with the
// element index innermost, in(blk,i,j,k,e) and G(blk,i,j,f,k,e), and the
// basis as basis(a,i) / dbasis(i,n) at row stride Q_pad (the next power of
// two >= nq).  The driver converts our arrays to that layout, runs the
// kernel, converts out back and compares with the reference.  nelmt is
// rounded down to a multiple of B.  DoFs: BK1, BK3 nelmt nm^3, BK5 nelmt nq^3;
// GB/s counts in + out + JxW (BK1) or in + out + G6 (BK3, BK5).

#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"

#ifndef TT_IR_DIR
#define TT_IR_DIR "tinytc/ir"
#endif

namespace {

int pad_pow2(const int n)
{
    int p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

// the work-group sizes of the original driver (tinytc.cc)
int default_wgs(const int nq)
{
    switch (nq) {
        case 2:
        case 3:
            return 32;
        case 4:
            return 64;
        case 5:
            return 128;
        case 6:
            return 256;
        case 7:
        case 8:
            return 512;
        default:
            return 32;
    }
}

sycl::kernel make_kernel(sycl::queue& q, const std::string& path, const char* name, const int nq, const int nq_pad,
                         const int B, const int sg, const int wgs)
{
    auto info = tinytc::create_core_info(q.get_device());
    // TT_FEATURES=n: the core feature flags requested (default: tinytc's choice)
    if (const auto f = bk::get_env("TT_FEATURES")) {
        tinytc::set_core_features(info.get(), static_cast<tinytc_core_feature_flags_t>(std::atoi(f->c_str())));
    }
    auto ctx = tinytc::create_compiler_context();
    tinytc::set_error_reporter(
        ctx.get(), [](char const* what, const tinytc_location_t*, void*) { std::cerr << what << std::endl; });
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    const std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::ostringstream code;
    code << "$Q = " << nq << "\n$Q_pad = " << nq_pad << "\n$B = " << B << "\n$M = " << sg << "\n$wgs = " << wgs
         << "\n" << body;
    auto prog = tinytc::parse_string(code.str(), ctx.get());
    auto bin = tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());
    std::fprintf(stderr, "tinytc: core features of the device %u, of the binary %u\n",
                 unsigned(tinytc::get_core_features(info.get())), unsigned(tinytc::get_core_features(bin.get())));
    // TT_SPV=path: write the SPIR-V (e.g. to compile it with ocloc for its build log)
    if (const auto spv = bk::get_env("TT_SPV")) {
        const auto raw = tinytc::get_raw(bin.get());
        std::ofstream(*spv, std::ios::binary).write(reinterpret_cast<const char*>(raw.data), raw.data_size);
    }
    auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), bin.get());
    return tinytc::create_kernel(bundle, name);
}

// our element-contiguous array of n points per element -> blocks of B
// elements with the element innermost
std::vector<float> to_blocks(const std::vector<float>& a, const std::size_t nelmt, const int n, const int B)
{
    std::vector<float> b(nelmt * n);
    for (std::size_t e = 0; e < nelmt; ++e) {
        const std::size_t blk = e / B;
        const std::size_t el = e % B;
        for (int x = 0; x < n; ++x) {
            b[(blk * n + x) * B + el] = a[e * n + x];
        }
    }
    return b;
}

std::vector<float> from_blocks(const std::vector<float>& b, const std::size_t nelmt, const int n, const int B)
{
    std::vector<float> a(nelmt * n);
    for (std::size_t e = 0; e < nelmt; ++e) {
        for (int x = 0; x < n; ++x) {
            a[e * n + x] = b[((e / B) * n + x) * B + e % B];
        }
    }
    return a;
}

template <typename T>
T* to_device(sycl::queue& q, const std::vector<T>& h)
{
    T* d = sycl::malloc_device<T>(h.size(), q);
    q.copy(h.data(), d, h.size()).wait_and_throw();
    return d;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <1|3|5> <p> [nelmt] [ntests]\n", argv[0]);
        return 2;
    }
    const int bk = std::atoi(argv[1]);
    const int p = std::atoi(argv[2]);
    const int nq = p + ((bk == 5) ? 1 : 2);
    const int B = bk::env_int("TT_B", 16);
    const int sg = bk::env_int("TT_SG", 16);
    std::size_t nelmt = (argc > 3) ? std::size_t(std::atoll(argv[3])) : std::size_t(1) << 17;
    nelmt = std::max<std::size_t>(B, nelmt / B * B);
    const int ntests = (argc > 4) ? std::atoi(argv[4]) : 20;
    const int wgs = bk::env_int("TT_WGS", default_wgs(nq));
    const std::string dir = bk::get_env("TT_IR_DIR").value_or(TT_IR_DIR);
    const std::string layout = bk::get_env("TT_LAYOUT").value_or("ec");
    const bool chain = (layout == "blk_ch");
    const bool blocked = (layout == "blk" || layout == "orig" || chain);
    if (!blocked && layout != "ec") {
        std::fprintf(stderr, "TT_LAYOUT=%s: ec, blk, blk_ch or orig\n", layout.c_str());
        return 2;
    }
    if (!blocked && B != 16 && bk != 3) {
        // the _ec kernels map a 16-row tile to one spatial pair of the block
        std::fprintf(stderr, "TT_LAYOUT=ec needs TT_B=16\n");
        return 2;
    }
    const int pad = bk::env_int("TT_QPAD", pad_pow2(nq));
    const int nm = nq - 1;
    if (bk != 1 && bk != 3 && bk != 5) {
        std::fprintf(stderr, "BK%d: only 1, 3 and 5\n", bk);
        return 2;
    }
    // BK3: the local arrays hold B nq^3 floats each, or B nm nq^2 chained (blk_ch)
    const int array_bytes = B * (chain ? nm : nq) * nq * nq * int(sizeof(float));
    const bool spill = (bk == 3) && bk::env_int("TT_SPILL", 3 * array_bytes > 65536) != 0;
    if (chain && (bk != 3 || spill)) {
        std::fprintf(stderr, "TT_LAYOUT=blk_ch: BK3 only, without TT_SPILL\n");
        return 2;
    }
    if (bk == 3) {
        if (layout == "orig" || B % sg != 0) {
            std::fprintf(stderr, "BK3: TT_LAYOUT ec, blk or blk_ch, TT_B a multiple of TT_SG\n");
            return 2;
        }
        // three arrays (two with S spilled) in the 64 KiB of local memory
        const int arrays = spill ? 2 : 3;
        if (arrays * array_bytes > 65536) {
            std::fprintf(stderr, "BK3 tinytc: %d arrays of %d bytes (B = %d, nq = %d) exceed 64 KiB of local memory\n",
                         arrays, array_bytes, B, nq);
            return 2;
        }
    }

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const int npts = (bk == 5) ? nq * nq * nq : nm * nm * nm;   // in/out values per element
    const int ngeo = nq * nq * nq * ((bk == 1) ? 1 : 6);         // JxW or G6 per element
    const std::string label = layout + (spill ? "_gs" : "");
    std::printf("BK%d tinytc (%s): p = %d, nq = %d, Q_pad = %d, B = %d, sg = %d, wgs = %d, nelmt = %zu (%zu DoFs), "
                "ntests = %d\n",
                bk, label.c_str(), p, nq, pad, B, sg, wgs, nelmt, nelmt * npts, ntests);

    // our data and reference
    std::vector<float> in(nelmt * npts);
    std::vector<float> geo(nelmt * ngeo);   // JxW, G6 factor-major (BK3) or interleaved (BK5)
    bk::fill_random_test_data(in, geo);
    std::vector<double> ref(in.size());
    std::vector<float> basis_t(pad * pad, 0.0f);   // tinytc's padded basis
    std::vector<float> dbasis_t(pad * pad, 0.0f);  // BK3: and derivative
    std::vector<float> bd_t(pad * pad, 0.0f);      // BK3 chained: B D
    if (bk == 3) {
        const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
        const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
        for (int i = 0; i < nm; ++i) {
            for (int a = 0; a < nq; ++a) {
                basis_t[a * pad + i] = basis[i * nq + a];   // Bm(i, a) = B(i, a)
            }
        }
        for (int n = 0; n < nq; ++n) {
            for (int a = 0; a < nq; ++a) {
                dbasis_t[a * pad + n] = D[n * nq + a];   // Dm(n, a) = D(n, a)
            }
        }
        for (int i = 0; i < nm; ++i) {
            for (int a = 0; a < nq; ++a) {
                double sum = 0.0;
                for (int n = 0; n < nq; ++n) {
                    sum += double(basis[i * nq + n]) * double(D[n * nq + a]);
                }
                bd_t[a * pad + i] = float(sum);   // BDm(i, a) = sum_n B(i, n) D(n, a)
            }
        }
        bk::reference_bk3(nq, nelmt, basis, D, geo, in, ref);
    } else if (bk == 1) {
        const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
        for (int i = 0; i < nm; ++i) {
            for (int a = 0; a < nq; ++a) {
                basis_t[a * pad + i] = basis[i * nq + a];   // basis0(a, i)
            }
        }
        bk::reference_bk1(nq, nelmt, basis, geo, in, ref);
    } else {
        const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
        for (int i = 0; i < nq; ++i) {
            for (int n = 0; n < nq; ++n) {
                basis_t[i * pad + n] = D[i * nq + n];   // dbasis(i, n)
            }
        }
        bk::reference_bk5(nq, nelmt, D, geo, in, ref);
    }
    const double big = bk::max_abs(ref);

    // blk: tinytc's layout, blocks of B elements, element innermost
    float* d_basis = to_device(q, basis_t);
    float* d_dbasis = to_device(q, dbasis_t);
    float* d_bd = to_device(q, bd_t);
    float* d_geo = blocked ? to_device(q, to_blocks(geo, nelmt, ngeo, B)) : to_device(q, geo);
    float* d_in = blocked ? to_device(q, to_blocks(in, nelmt, npts, B)) : to_device(q, in);
    float* d_out = sycl::malloc_device<float>(in.size(), q);
    // BK3 with TT_SPILL: S, B nq^3 floats per block
    float* d_scratch = spill ? sycl::malloc_device<float>(nelmt * nq * nq * nq, q) : nullptr;
    q.memset(d_out, 0, in.size() * sizeof(float)).wait();
    geo = std::vector<float>();

    std::string ir = "/bk" + std::to_string(bk);
    if (layout == "ec") {
        ir += "_ec";
    } else if (layout == "blk" && bk != 1) {
        ir += "_blk";
    } else if (chain) {
        ir += "_blk_ch";
    }
    ir += spill ? "_gs.ir" : ".ir";
    sycl::kernel k = [&]() {
        try {
            return make_kernel(q, dir + ir, ("bk" + std::to_string(bk)).c_str(), nq, pad, B, sg, wgs);
        } catch (const tinytc::status& st) {
            std::fprintf(stderr, "tinytc error %d: %s\n", static_cast<int>(st), tinytc::to_string(st));
            std::exit(1);
        }
    }();
    const auto range = tinytc::get_execution_range(k, sycl::range<3u>{1u, 1u, nelmt / B});
    const std::int64_t n64 = static_cast<std::int64_t>(nelmt / B);
    const auto launch = [&]() {
        return q.submit([&](sycl::handler& h) {
            if (spill) {
                h.set_args(d_basis, d_dbasis, d_geo, n64, d_in, n64, d_out, n64, d_scratch, n64);
            } else if (chain) {
                h.set_args(d_basis, d_dbasis, d_bd, d_geo, n64, d_in, n64, d_out, n64);
            } else if (bk == 3) {
                h.set_args(d_basis, d_dbasis, d_geo, n64, d_in, n64, d_out, n64);
            } else {
                h.set_args(d_basis, d_basis, d_basis, d_geo, n64, d_in, n64, d_out, n64);
            }
            h.parallel_for(range, k);
        });
    };
    const bk::launch_times t = bk::time_launches(ntests, launch);
    std::vector<float> out(in.size());
    q.copy(d_out, out.data(), out.size()).wait();
    if (blocked) {
        out = from_blocks(out, nelmt, npts, B);
    }
    const double relerr = bk::rel_maxdiff(out, ref, big);
    const double dofs = double(nelmt) * npts;
    const double bytes = sizeof(float) * double(nelmt) * (2.0 * npts + ngeo);
    std::printf("tt%d_%-10s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s\n", bk,
                label.c_str(), 1e-9 * dofs / t.host, 1e-9 * bytes / t.host, 1e-9 * dofs / t.device, 1e-9 * bytes / t.device,
                relerr, bk::fail_mark(relerr));
    sycl::free(d_basis, q);
    sycl::free(d_dbasis, q);
    sycl::free(d_bd, q);
    sycl::free(d_geo, q);
    sycl::free(d_in, q);
    sycl::free(d_out, q);
    if (d_scratch != nullptr) {
        sycl::free(d_scratch, q);
    }
    return 0;
}
