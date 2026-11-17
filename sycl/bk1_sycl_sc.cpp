// bk1_sycl_sc.cpp -- BK1 in SYCL, every order and kernel shape in one binary:
// the opt kernel of sycl/bk1_sycl_opt.h with nq, the batch, the work-group
// size and the SLM layout from a specialization constant.  Each selected
// shape is JIT-compiled with them as constants before the runs
// (bk::build_specialized), and its launches use that kernel bundle.
//
//   scripts/build.sh bk1_sycl_sc
//   scripts/run.sh bk1_sycl_sc <p> [nelmt] [ntests]     (nq = p + 2)
//   BK_KERNEL="scP_b1 scF_f128" ...      run a subset (default: all)
//   BK_SHAPE="b4w256p1 b1w64io1" ...      run these shapes instead: b batch,
//                                         w work-group size, p0 dense or p1
//                                         padded SLM, io1 sub-group block copies
//   BK_SHAPE_FILE=<file>                  the same from the lines "<p> <shape>..."
//                                         of a file for this p (# comments)
//
// The default shapes are those of bk1_sycl_<p> (bk1opt::opt_variants), named
// sc* for inl* (inlP_b1 -> scP_b1).  The result lines are those of bk1_sycl,
// each followed by the shape's line of sycl/README.md (convention 4), and a
// "JIT:" line before the runs sums up the builds.  Data and check:
// sycl/bk1_sycl_harness.h.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"
#include "bk1_sycl_harness.h"
#include "bk1_sycl_opt.h"

namespace {

using bk1opt::opt_params;
using exe_bundle = sycl::kernel_bundle<sycl::bundle_state::executable>;

struct shape {
    std::string name;
    opt_params P;
    bool block_io;
};

// "b<batch>w<block>p<slm>[io1]"
std::string shape_word(const opt_params& P, const bool block_io)
{
    return "b" + std::to_string(P.batch) + "w" + std::to_string(P.block) + "p" + std::to_string(P.slm) +
           (block_io ? "io1" : "");
}

// the bk1_sycl_<p> variant of the same shape, "-" if none
const char* aot_twin(const opt_params& P, const bool block_io)
{
    for (const bk1opt::opt_variant& v : bk1opt::opt_variants) {
        if (v.block_io == block_io && bk1opt::same_shape(bk1opt::variant_params(P.nq, v), P)) {
            return v.aot;
        }
    }
    return "-";
}

// a shape word b<batch>w<block>[p<0|1>][io<0|1>], in that order
bool parse_shape(const std::string& w, const int nq, shape& s)
{
    const char* c = w.c_str();
    int batch = 0;
    int block = 0;
    int n = 0;
    int m = 0;
    int p = 0;
    int io = 0;
    if (std::sscanf(c, "b%dw%d%n", &batch, &block, &n) != 2) {
        return false;
    }
    if (std::sscanf(c + n, "p%d%n", &p, &m) == 1) {
        n += m;
    }
    if (std::sscanf(c + n, "io%d%n", &io, &m) == 1) {
        n += m;
    }
    s = {w, bk1opt::make_opt_params(nq, batch, block, p), io != 0};
    return c[n] == '\0' && (p == bk1opt::dense || p == bk1opt::padded);
}

// the shapes to run: BK_SHAPE, else the lines of BK_SHAPE_FILE for this p,
// else bk1opt::opt_variants; false on a word that does not parse
bool selected_shapes(const int nq, std::vector<shape>& shapes)
{
    std::string words = bk::get_env("BK_SHAPE").value_or("");
    if (words.empty()) {
        if (const auto file = bk::get_env("BK_SHAPE_FILE")) {
            std::ifstream in(*file);
            if (!in) {
                std::fprintf(stderr, "BK_SHAPE_FILE: cannot read %s\n", file->c_str());
                return false;
            }
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line.substr(0, line.find('#')));
                int p = -1;
                if ((ls >> p) && p == nq - 2) {
                    std::string w;
                    while (ls >> w) {
                        words += " " + w;
                    }
                }
            }
            if (words.empty()) {
                std::fprintf(stderr, "BK_SHAPE_FILE: no shapes for p = %d in %s\n", nq - 2, file->c_str());
                return false;
            }
        }
    }
    if (words.empty()) {
        for (const bk1opt::opt_variant& v : bk1opt::opt_variants) {
            shapes.push_back({v.sc, bk1opt::variant_params(nq, v), v.block_io});
        }
        return true;
    }
    std::istringstream ss(words);
    std::string w;
    while (ss >> w) {
        shape s;
        if (!parse_shape(w, nq, s)) {
            std::fprintf(stderr, "cannot parse shape \"%s\" (b<batch>w<block>[p<0|1>][io<0|1>])\n", w.c_str());
            return false;
        }
        shapes.push_back(s);
    }
    return true;
}

// the variant of shape s: built for it here, launched with that bundle
template <bool block_io>
bk1sycl::variant make_variant(sycl::queue& q, const shape& s, double& build_total)
{
    const bk::specialized_kernel k =
        bk::build_specialized<bk1opt::sc_kernel<float, block_io>, bk1opt::opt_id>(q, s.P);
    build_total += k.seconds;
    char secs[32];
    std::snprintf(secs, sizeof(secs), " build %.3f s", k.seconds);
    const opt_params P = s.P;
    const exe_bundle exe = k.bundle;
    return {s.name,
            [P, exe](sycl::queue& q, const bk1sycl::device_data& d) {
                return bk1opt::launch<float, bk1opt::sc_shape, block_io>(q, P, d, &exe);
            },
            "shape " + shape_word(s.P, block_io) + " aot " + aot_twin(s.P, block_io) + secs};
}

} // namespace

int main(int argc, char** argv)
{
    const int nq = (argc > 1) ? std::atoi(argv[1]) + 2 : 0;
    if (nq < 2 || nq > MAXNQ) {
        std::fprintf(stderr, "usage: bk1_sycl_sc <p> [nelmt] [ntests], p = 0..%d (MAXNQ - 2)\n", MAXNQ - 2);
        return 2;
    }
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : bk::default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::vector<shape> shapes;
    if (!selected_shapes(nq, shapes)) {
        return 2;
    }
    sycl::queue q = bk::make_queue();
    std::vector<bk1sycl::variant> run;
    double build_total = 0;
    for (const shape& s : shapes) {
        if (!bk::selected(s.name.c_str())) {
            continue;
        }
        if (!bk1opt::opt_valid(s.P, s.block_io)) {
            std::printf("%-14s skipped: batch %d, block %d (one work-item per point of the widest step, whole "
                        "sub-groups of 16 for block copies)\n",
                        s.name.c_str(), s.P.batch, s.P.block);
            continue;
        }
        try {
            run.push_back(s.block_io ? make_variant<true>(q, s, build_total) : make_variant<false>(q, s, build_total));
        } catch (const sycl::exception& e) {
            std::printf("%-14s skipped: %s\n", s.name.c_str(), e.what());
        }
    }
    std::printf("JIT: %zu shapes built in %.2f s (SYCL_CACHE_PERSISTENT=%s)\n", run.size(), build_total,
                bk::get_env("SYCL_CACHE_PERSISTENT").value_or("0").c_str());
    bk1sycl::run(q, "BK1 SYCL sc", nq, nelmt, ntests, run);
    return 0;
}
