// logspace -- print N values spaced evenly in log10 over [lo, hi], inclusive,
// each rounded to the nearest integer, one per line.
//
// Used by run_benchmarks.sh to pick logarithmically spaced DoF targets. It is
// deliberately plain ISO C++ (no OpenMP, no target offload, no awk dialect
// quirks) so it compiles portably wherever a C++ compiler exists.
//
//   usage: logspace <lo> <hi> <n>
//   e.g.:  logspace 1e4 1e8 5   ->   10000 / 100000 / 1000000 / 10000000 / 100000000

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <lo> <hi> <n>\n", argv[0]);
        return 1;
    }

    const double lo = std::atof(argv[1]);
    const double hi = std::atof(argv[2]);
    const int    n  = std::atoi(argv[3]);

    if (lo <= 0.0 || hi <= 0.0) {
        std::fprintf(stderr, "logspace: lo and hi must be positive\n");
        return 1;
    }
    if (n < 1) {
        std::fprintf(stderr, "logspace: n must be >= 1\n");
        return 1;
    }

    const double llo = std::log10(lo);
    const double lhi = std::log10(hi);

    for (int i = 0; i < n; ++i) {
        const double e = (n == 1) ? llo : llo + (lhi - llo) * i / (n - 1);
        std::printf("%.0f\n", std::pow(10.0, e));
    }
    return 0;
}
