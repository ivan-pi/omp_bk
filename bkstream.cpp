#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstddef>
#include <cctype>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>   // getopt

#include "bk_common.h"

// ---------------------------------------------------------------------------
// bkstream: STREAM-style memory-bandwidth micro-benchmarks that reuse the BK
// degree-of-freedom (DoF) data layout -- a flat std::vector of shape
// nelmt * nm * nm * nm (nm = p + 1 modes per direction) -- and the same
// `omp target ... teams loop` construct and min-over-repetitions timing as the
// BK1/BK3/BK5 kernels. The point is a peak effective-bandwidth reference
// obtained under a setup directly comparable to those kernels.
//
// Kernels (Y is written, X/Z/W are read; `a` is a scalar constant):
//   init    Y = a                    1 array
//   copy    Y = X                    2 arrays
//   triad   Y = X + a * Z            3 arrays
//   striad  Y = X + Z * W            4 arrays   (Schoenauer triad)
//
// Byte traffic is counted as (#arrays) * ndof * sizeof(T) per repetition.
//
// Caveat: unlike the BK operators these kernels do not stream a geometric
// factor array (JxW / G). On a uniform grid that array degenerates to a single
// element that would be broadcast, contributing negligible traffic, so it is
// omitted here; expect these numbers to sit somewhat above the BK effective
// bandwidth, which does pay for reading its metric arrays.
// ---------------------------------------------------------------------------

namespace bk {

enum class Kernel { Init, Copy, Triad, Striad };

inline const char* kernel_name(Kernel k)
{
    switch (k) {
        case Kernel::Init:   return "init";
        case Kernel::Copy:   return "copy";
        case Kernel::Triad:  return "triad";
        case Kernel::Striad: return "striad";
    }
    return "?";
}

// Number of distinct arrays streamed, used for the byte-traffic count.
inline int kernel_arrays(Kernel k)
{
    switch (k) {
        case Kernel::Init:   return 1;
        case Kernel::Copy:   return 2;
        case Kernel::Triad:  return 3;
        case Kernel::Striad: return 4;
    }
    return 0;
}

using hrc = std::chrono::high_resolution_clock;

// Each kernel runs inside an enclosing `target data` region so the arrays are
// resident on the device for the whole repetition loop; the per-repetition
// `target ... teams loop` then re-states the same maps, which resolve to
// present (no transfer) exactly as in the BK kernels. The minimum wall time
// over the repetitions is returned.

// Time `launch` over ntests repetitions and return the minimum wall time (s).
template <typename Launch>
double timed_min(int ntests, Launch&& launch)
{
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < ntests; ++t) {
        auto s = hrc::now();
        launch();
        best = std::min(best, std::chrono::duration<double>(hrc::now() - s).count());
    }
    return best;
}

template <typename T>
double run_init(std::size_t nelmt, int ND, T* Y, T a, int ntests)
{
    const std::size_t n = nelmt * std::size_t(ND);
    double best;
    #pragma omp target data map(from: Y[:n])
    best = timed_min(ntests, [&] {
        #pragma omp target map(from: Y[:n])
        #pragma omp teams loop
        for (std::size_t e = 0; e < nelmt; ++e)
            for (int i = 0; i < ND; ++i)
                Y[e * ND + i] = a;
    });
    return best;
}

template <typename T>
double run_copy(std::size_t nelmt, int ND, const T* X, T* Y, int ntests)
{
    const std::size_t n = nelmt * std::size_t(ND);
    double best;
    #pragma omp target data map(to: X[:n]) map(from: Y[:n])
    best = timed_min(ntests, [&] {
        #pragma omp target map(to: X[:n]) map(from: Y[:n])
        #pragma omp teams loop
        for (std::size_t e = 0; e < nelmt; ++e)
            for (int i = 0; i < ND; ++i)
                Y[e * ND + i] = X[e * ND + i];
    });
    return best;
}

template <typename T>
double run_triad(std::size_t nelmt, int ND, const T* X, const T* Z, T* Y,
                 T a, int ntests)
{
    const std::size_t n = nelmt * std::size_t(ND);
    double best;
    #pragma omp target data map(to: X[:n], Z[:n]) map(from: Y[:n])
    best = timed_min(ntests, [&] {
        #pragma omp target map(to: X[:n], Z[:n]) map(from: Y[:n])
        #pragma omp teams loop
        for (std::size_t e = 0; e < nelmt; ++e)
            for (int i = 0; i < ND; ++i)
                Y[e * ND + i] = X[e * ND + i] + a * Z[e * ND + i];
    });
    return best;
}

template <typename T>
double run_striad(std::size_t nelmt, int ND, const T* X, const T* Z,
                  const T* W, T* Y, int ntests)
{
    const std::size_t n = nelmt * std::size_t(ND);
    double best;
    #pragma omp target data map(to: X[:n], Z[:n], W[:n]) map(from: Y[:n])
    best = timed_min(ntests, [&] {
        #pragma omp target map(to: X[:n], Z[:n], W[:n]) map(from: Y[:n])
        #pragma omp teams loop
        for (std::size_t e = 0; e < nelmt; ++e)
            for (int i = 0; i < ND; ++i)
                Y[e * ND + i] = X[e * ND + i] + Z[e * ND + i] * W[e * ND + i];
    });
    return best;
}

// Dispatch one kernel at a given problem size and return the min time.
template <typename T>
double run_kernel(Kernel k, std::size_t nelmt, int ND, int ntests)
{
    const std::size_t n = nelmt * std::size_t(ND);
    const T a = T(3.0);
    std::vector<T> Y(n, T(0));

    switch (k) {
        case Kernel::Init:
            return run_init<T>(nelmt, ND, Y.data(), a, ntests);
        case Kernel::Copy: {
            std::vector<T> X(n, T(1));
            return run_copy<T>(nelmt, ND, X.data(), Y.data(), ntests);
        }
        case Kernel::Triad: {
            std::vector<T> X(n, T(1)), Z(n, T(2));
            return run_triad<T>(nelmt, ND, X.data(), Z.data(), Y.data(), a, ntests);
        }
        case Kernel::Striad: {
            std::vector<T> X(n, T(1)), Z(n, T(2)), W(n, T(0.5));
            return run_striad<T>(nelmt, ND, X.data(), Z.data(), W.data(),
                                 Y.data(), ntests);
        }
    }
    return 0.0;
}

// Parse a size string: a number with an optional binary suffix (B, K/KB,
// M/MB, G/GB; base 1024, case-insensitive). A bare number is bytes; scientific
// notation is accepted (e.g. "1e6").
std::size_t parse_size(const std::string& s)
{
    std::size_t pos = 0;
    double val;
    try {
        val = std::stod(s, &pos);
    } catch (const std::exception&) {
        throw std::runtime_error("cannot parse size '" + s + "'");
    }
    std::string suf = s.substr(pos);
    // strip spaces and upper-case
    std::string u;
    for (char c : suf)
        if (!std::isspace(static_cast<unsigned char>(c)))
            u += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    double mult;
    if (u.empty() || u == "B")        mult = 1.0;
    else if (u == "K" || u == "KB")   mult = 1024.0;
    else if (u == "M" || u == "MB")   mult = 1024.0 * 1024.0;
    else if (u == "G" || u == "GB")   mult = 1024.0 * 1024.0 * 1024.0;
    else throw std::runtime_error("unknown size suffix in '" + s + "'");

    if (val <= 0.0)
        throw std::runtime_error("size must be positive: '" + s + "'");
    return static_cast<std::size_t>(std::llround(val * mult));
}

} // namespace bk

using namespace bk;

static void usage(const char* prog, std::ostream& os = std::cout)
{
    os <<
"Usage: " << prog << " [options] [kernel]\n"
"\n"
"STREAM-style bandwidth benchmarks over the BK DoF layout\n"
"(std::vector of nelmt*nm*nm*nm, nm = p + 1), swept over a logarithmic\n"
"range of per-array sizes. Output is a gnuplot data file (see scripts/).\n"
"\n"
"  kernel       One of: init copy triad striad all   (default: all).\n"
"               init=Y=a  copy=Y=X  triad=Y=X+a*Z  striad=Y=X+Z*W.\n"
"\n"
"Options:\n"
"  -r MIN:MAX   Per-array size range, log-spaced. Accepts B/K/M/G suffixes\n"
"               (base 1024) or bare bytes / scientific   (default 1K:64M).\n"
"  -n N         Number of log-spaced sample points             (default 12).\n"
"  -p P         Polynomial order; layout uses nm = P + 1       (default 4).\n"
"  -t NTESTS    Timing repetitions; the minimum is reported    (default 5).\n"
"  -h           Show this help and exit.\n"
"\n"
"Example (write a data file, then plot it):\n"
"  " << prog << " -r 1K:256M all > results/bkstream.dat\n"
"  gnuplot -c scripts/plot_stream.gp results/bkstream.dat\n";
}

int main(int argc, char** argv)
{
    using T = float;

    std::string range = "1K:64M";
    int npoints = 12;
    int p       = 4;
    int ntests  = 5;

    int opt;
    while ((opt = getopt(argc, argv, "r:n:p:t:h")) != -1) {
        switch (opt) {
            case 'r': range   = optarg; break;
            case 'n': npoints = std::atoi(optarg); break;
            case 'p': p       = std::atoi(optarg); break;
            case 't': ntests  = std::atoi(optarg); break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0], std::cerr); return 1;
        }
    }

    const std::string kern = (optind < argc) ? argv[optind] : "all";

    // Select kernels (one block per kernel in the output).
    std::vector<Kernel> kernels;
    if      (kern == "all")    kernels = {Kernel::Init, Kernel::Copy,
                                          Kernel::Triad, Kernel::Striad};
    else if (kern == "init")   kernels = {Kernel::Init};
    else if (kern == "copy")   kernels = {Kernel::Copy};
    else if (kern == "triad")  kernels = {Kernel::Triad};
    else if (kern == "striad") kernels = {Kernel::Striad};
    else {
        std::cerr << "error: unknown kernel '" << kern
                  << "' (init|copy|triad|striad|all)\n";
        return 1;
    }

    if (p < 1)       { std::cerr << "error: p must be >= 1\n";       return 1; }
    if (npoints < 1) { std::cerr << "error: -n must be >= 1\n";      return 1; }
    if (ntests < 1)  { std::cerr << "error: -t must be >= 1\n";      return 1; }

    // Split the range on ':'.
    const auto colon = range.find(':');
    if (colon == std::string::npos) {
        std::cerr << "error: range must be MIN:MAX, got '" << range << "'\n";
        return 1;
    }
    std::size_t smin, smax;
    try {
        smin = parse_size(range.substr(0, colon));
        smax = parse_size(range.substr(colon + 1));
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    if (smin > smax) std::swap(smin, smax);

    const int nm = p + 1;
    const int ND = nm * nm * nm;

    // Log-spaced per-array byte targets -> element counts (deduplicated).
    const double lmin = std::log10(double(smin));
    const double lmax = std::log10(double(smax));
    std::vector<std::size_t> nelems;
    long prev = -1;
    for (int i = 0; i < npoints; ++i) {
        const double e = (npoints == 1) ? lmin
                                        : lmin + (lmax - lmin) * i / (npoints - 1);
        const double bytes = std::pow(10.0, e);
        const double ndof  = bytes / double(sizeof(T));
        long nel = std::lround(ndof / double(ND));
        if (nel < 1) nel = 1;
        if (nel == prev) continue;
        prev = nel;
        nelems.push_back(std::size_t(nel));
    }

    // --- gnuplot data-file header ------------------------------------------
    std::cout << "# benchmark = bkstream (BK DoF layout, T=float)\n"
              << "# key = kernel\n"
              << "# p = " << p << "  nm = " << nm << "  ND = nm^3 = " << ND << "\n"
              << "# repetitions = " << ntests << "\n";
    std::cout << "# kernels =";
    for (Kernel k : kernels) std::cout << ' ' << kernel_name(k);
    std::cout << "\n# columns: nelmt  ndof  bytes  time_s  gbytes_per_s\n";

    std::cerr << "bkstream: kernels[" << kernels.size() << "]  p=" << p
              << " (ND=" << ND << ")  range " << range
              << "  npoints=" << npoints << "  ntests=" << ntests << "\n";

    std::cout << std::setprecision(8);

    for (std::size_t bi = 0; bi < kernels.size(); ++bi) {
        const Kernel k = kernels[bi];
        const int arrays = kernel_arrays(k);

        if (bi > 0) std::cout << "\n\n";     // gnuplot index separator
        std::cout << "\n# kernel = " << kernel_name(k)
                  << "  (arrays = " << arrays << ")\n";

        for (std::size_t nelmt : nelems) {
            const std::size_t ndof  = nelmt * std::size_t(ND);
            const std::size_t bytes = ndof * sizeof(T);
            const double time = run_kernel<T>(k, nelmt, ND, ntests);
            const double gbs  = 1.0e-9 * double(arrays) * double(bytes) / time;

            std::cout << nelmt << ' ' << ndof << ' ' << bytes << ' '
                      << time << ' ' << gbs << '\n';
        }
        std::cerr << "  " << kernel_name(k) << ": done\n";
    }

    return 0;
}
