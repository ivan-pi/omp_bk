// bk_tinytc.cpp -- BK1 and BK5 kernels written for the Intel Tiny Tensor
// Compiler (tinytc), run on the data and checked against the
// double-precision references of this repository's BK1 and BK5 harnesses
// (sycl/bk1_sycl.cpp, openmp/offload/bk5_offload.cpp).
//
// TT_LAYOUT=ec (default): tinytc/ir/bk1_ec.ir and bk5_ec.ir, the kernels
// rewritten for our element-contiguous arrays, which they take unchanged.
// TT_LAYOUT=blk: the original kernels, tinytc/ir/bk1.ir and bk5_blk.ir
// (bk5.ir with its global-memory race fixed), on our arrays converted to
// their blocked layout (below) and the result converted back.
// TT_LAYOUT=orig: as blk with the unmodified bk5.ir.
//
//   scripts/build.sh bk_tinytc              (-> build/a770/bk_tinytc)
//   build/a770/bk_tinytc <1|5> <nq> [nelmt] [ntests]
//   env: TT_LAYOUT (ec or blk), TT_IR_DIR (default: tinytc/ir of the source tree),
//        TT_QPAD (row stride of the basis, default: the next power of two >= nq),
//        TT_FEATURES (tinytc core feature flags, e.g. 0 for none),
//        TT_SPV (write the generated SPIR-V to this file),
//        TT_B (elements per block, default 16), TT_WGS (work-group size,
//        default: the table of the original driver)
//
// Conventions.  Ours: element-contiguous in/out (BK1 in(e,i,j,k) of nm^3
// modes, BK5 in(e,i,j,k) of nq^3 points), BK1 JxW(e,p,q,r), BK5 G6
// interleaved G(e,i,j,f,k), BK1 basis(i,p) = cos(i nq + p), BK5 D(i,n) =
// cos(i nq + n).  tinytc's kernels take blocks of B elements with the
// element index innermost, in(blk,i,j,k,e) and G(blk,i,j,f,k,e), and the
// basis as basis(a,i) / dbasis(i,n) at row stride Q_pad (the next power of
// two >= nq).  The driver converts our arrays to that layout, runs the
// kernel, converts out back and compares with the reference.  nelmt is
// rounded down to a multiple of B.  DoFs: BK1 nelmt nm^3, BK5 nelmt nq^3;
// GB/s counts in + out + JxW (BK1) or in + out + G6 (BK5).

#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"

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
                         const int B, const int wgs)
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
    code << "$Q = " << nq << "\n$Q_pad = " << nq_pad << "\n$B = " << B << "\n$wgs = " << wgs << "\n" << body;
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

void reference_bk1(const int nq, const std::size_t nelmt, const std::vector<float>& basis,
                   const std::vector<float>& JxW, const std::vector<float>& in, std::vector<double>& out)
{
    const int nm = nq - 1;
    std::vector<double> w0(nq * nq * nq);
    std::vector<double> w1(nq * nq * nq);
    std::vector<double> w2(nq * nq * nq);
    for (std::size_t e = 0; e < nelmt; ++e) {
        const float* ei = in.data() + e * nm * nm * nm;
        const float* ej = JxW.data() + e * nq * nq * nq;
        for (int p = 0; p < nq; ++p) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int i = 0; i < nm; ++i) {
                        s += double(ei[(i * nm + j) * nm + k]) * basis[i * nq + p];
                    }
                    w0[(p * nm + j) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int j = 0; j < nm; ++j) {
                        s += w0[(p * nm + j) * nm + k] * basis[j * nq + q];
                    }
                    w1[(p * nq + q) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    double s = 0;
                    for (int k = 0; k < nm; ++k) {
                        s += w1[(p * nq + q) * nm + k] * basis[k * nq + r];
                    }
                    w2[(p * nq + q) * nq + r] = s * ej[(p * nq + q) * nq + r];
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int r = 0; r < nq; ++r) {
                        s += w2[(p * nq + q) * nq + r] * basis[k * nq + r];
                    }
                    w1[(p * nq + q) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int q = 0; q < nq; ++q) {
                        s += w1[(p * nq + q) * nm + k] * basis[j * nq + q];
                    }
                    w0[(p * nm + j) * nm + k] = s;
                }
            }
        }
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int p = 0; p < nq; ++p) {
                        s += w0[(p * nm + j) * nm + k] * basis[i * nq + p];
                    }
                    out[e * nm * nm * nm + (i * nm + j) * nm + k] = s;
                }
            }
        }
    }
}

void reference_bk5(const int nq, const std::size_t nelmt, const std::vector<float>& D, const std::vector<float>& G6i,
                   const std::vector<float>& in, std::vector<double>& out)
{
    const int nq3 = nq * nq * nq;
    std::vector<double> rr(nq3);
    std::vector<double> rs(nq3);
    std::vector<double> rt(nq3);
    for (std::size_t e = 0; e < nelmt; ++e) {
        const float* ei = in.data() + e * nq3;
        const float* eG = G6i.data() + e * 6 * nq3;
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    double qr = 0;
                    double qs = 0;
                    double qt = 0;
                    for (int n = 0; n < nq; ++n) {
                        qr += double(ei[(n * nq + j) * nq + k]) * D[i * nq + n];
                        qs += double(ei[(i * nq + n) * nq + k]) * D[j * nq + n];
                        qt += double(ei[(i * nq + j) * nq + n]) * D[k * nq + n];
                    }
                    const auto g = [&](const int f) { return double(eG[((i * nq + j) * 6 + f) * nq + k]); };
                    const int x = (i * nq + j) * nq + k;
                    rr[x] = g(0) * qr + g(1) * qs + g(2) * qt;
                    rs[x] = g(1) * qr + g(3) * qs + g(4) * qt;
                    rt[x] = g(2) * qr + g(4) * qs + g(5) * qt;
                }
            }
        }
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    double s = 0;
                    for (int n = 0; n < nq; ++n) {
                        s += rr[(n * nq + j) * nq + k] * D[n * nq + i];
                        s += rs[(i * nq + n) * nq + k] * D[n * nq + j];
                        s += rt[(i * nq + j) * nq + n] * D[n * nq + k];
                    }
                    out[e * nq3 + (i * nq + j) * nq + k] = s;
                }
            }
        }
    }
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
        std::fprintf(stderr, "usage: %s <1|5> <nq> [nelmt] [ntests]\n", argv[0]);
        return 2;
    }
    const int bk = std::atoi(argv[1]);
    const int nq = std::atoi(argv[2]);
    const int B = std::atoi(bk::get_env("TT_B").value_or("16").c_str());
    std::size_t nelmt = (argc > 3) ? std::size_t(std::atoll(argv[3])) : std::size_t(1) << 17;
    nelmt = std::max<std::size_t>(B, nelmt / B * B);
    const int ntests = (argc > 4) ? std::atoi(argv[4]) : 20;
    const int wgs = std::atoi(bk::get_env("TT_WGS").value_or(std::to_string(default_wgs(nq))).c_str());
    const std::string dir = bk::get_env("TT_IR_DIR").value_or(TT_IR_DIR);
    const std::string layout = bk::get_env("TT_LAYOUT").value_or("ec");
    const bool blocked = (layout == "blk" || layout == "orig");
    if (!blocked && layout != "ec") {
        std::fprintf(stderr, "TT_LAYOUT=%s: ec, blk or orig\n", layout.c_str());
        return 2;
    }
    if (!blocked && B != 16) {
        // the _ec kernels map a 16-row tile to one spatial pair of the block
        std::fprintf(stderr, "TT_LAYOUT=ec needs TT_B=16\n");
        return 2;
    }
    const int pad = std::atoi(bk::get_env("TT_QPAD").value_or(std::to_string(pad_pow2(nq))).c_str());
    const int nm = nq - 1;
    if (bk != 1 && bk != 5) {
        std::fprintf(stderr, "BK%d: only 1 and 5\n", bk);
        return 2;
    }

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const int npts = (bk == 1) ? nm * nm * nm : nq * nq * nq;   // in/out values per element
    std::printf("BK%d tinytc (%s): p = %d, nq = %d, Q_pad = %d, B = %d, wgs = %d, nelmt = %zu (%zu DoFs), ntests = %d\n",
                bk, layout.c_str(), (bk == 1) ? nq - 2 : nq - 1, nq, pad, B, wgs, nelmt, nelmt * npts, ntests);

    // our data and reference
    std::vector<float> in(nelmt * npts);
    std::vector<float> geo(nelmt * nq * nq * nq * ((bk == 1) ? 1 : 6));   // JxW or G6 interleaved
    bk::fill_random_test_data(in, geo);
    std::vector<double> ref(in.size());
    std::vector<float> basis_t(pad * pad, 0.0f);   // tinytc's padded basis
    if (bk == 1) {
        std::vector<float> basis(nm * nq);
        for (int i = 0; i < nm; ++i) {
            for (int p = 0; p < nq; ++p) {
                basis[i * nq + p] = std::cos(float(i * nq + p));
                basis_t[p * pad + i] = basis[i * nq + p];   // basis0(a = p, i)
            }
        }
        reference_bk1(nq, nelmt, basis, geo, in, ref);
    } else {
        std::vector<float> D(nq * nq);
        for (int i = 0; i < nq; ++i) {
            for (int n = 0; n < nq; ++n) {
                D[i * nq + n] = std::cos(float(i * nq + n));
                basis_t[i * pad + n] = D[i * nq + n];   // dbasis(i, n)
            }
        }
        reference_bk5(nq, nelmt, D, geo, in, ref);
    }
    double big = 0.0;
    for (const double v : ref) {
        big = std::max(big, std::fabs(v));
    }

    // blk: tinytc's layout, blocks of B elements, element innermost
    const int ngeo = nq * nq * nq * ((bk == 1) ? 1 : 6);
    float* d_basis = to_device(q, basis_t);
    float* d_geo = to_device(q, blocked ? to_blocks(geo, nelmt, ngeo, B) : geo);
    float* d_in = to_device(q, blocked ? to_blocks(in, nelmt, npts, B) : in);
    float* d_out = sycl::malloc_device<float>(in.size(), q);
    q.memset(d_out, 0, in.size() * sizeof(float)).wait();
    geo = std::vector<float>();

    std::string ir = (bk == 1) ? "/bk1" : "/bk5";
    if (layout == "ec") {
        ir += "_ec.ir";
    } else if (layout == "blk" && bk == 5) {
        ir += "_blk.ir";
    } else {
        ir += ".ir";
    }
    sycl::kernel k = [&]() {
        try {
            return make_kernel(q, dir + ir, (bk == 1) ? "bk1" : "bk5", nq, pad, B, wgs);
        } catch (const tinytc::status& st) {
            std::fprintf(stderr, "tinytc error %d: %s\n", static_cast<int>(st), tinytc::to_string(st));
            std::exit(1);
        }
    }();
    const auto range = tinytc::get_execution_range(k, sycl::range<3u>{1u, 1u, nelmt / B});
    const std::int64_t n64 = static_cast<std::int64_t>(nelmt / B);
    const auto launch = [&]() {
        return q.submit([&](sycl::handler& h) {
            h.set_args(d_basis, d_basis, d_basis, d_geo, n64, d_in, n64, d_out, n64);
            h.parallel_for(range, k);
        });
    };
    launch().wait_and_throw();   // warm-up
    double host_best = std::numeric_limits<double>::max();
    double dev_best = std::numeric_limits<double>::max();
    for (int t = 0; t < ntests; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        sycl::event ev = launch();
        ev.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        host_best = std::min(host_best, std::chrono::duration<double>(t1 - t0).count());
        dev_best = std::min(dev_best, 1e-9 * double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                    ev.get_profiling_info<sycl::info::event_profiling::command_start>()));
    }
    std::vector<float> out_b(in.size());
    q.copy(d_out, out_b.data(), out_b.size()).wait();
    const std::vector<float> out = blocked ? from_blocks(out_b, nelmt, npts, B) : out_b;
    double diff = 0.0;
    for (std::size_t x = 0; x < out.size(); ++x) {
        diff = std::max(diff, std::fabs(double(out[x]) - ref[x]));
    }
    const double dofs = double(nelmt) * npts;
    const double bytes = sizeof(float) * double(nelmt) * (2.0 * npts + ngeo);
    std::printf("tt%d_%-10s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s\n", bk,
                layout.c_str(), 1e-9 * dofs / host_best, 1e-9 * bytes / host_best, 1e-9 * dofs / dev_best, 1e-9 * bytes / dev_best,
                diff / big, (diff / big < 1e-5) ? "" : "  FAIL");
    sycl::free(d_basis, q);
    sycl::free(d_geo, q);
    sycl::free(d_in, q);
    sycl::free(d_out, q);
    return 0;
}
