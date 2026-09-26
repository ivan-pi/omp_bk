// amx_pipe.cpp -- how the AMX unit's throughput depends on the distance
// between a load and the outer product that consumes it, and on stores.
//
//   make amx_pipe      (or: clang++ -O3 -mcpu=native -std=c++17 amx/bench/amx_pipe.cpp -o amx_pipe)
//   ./amx_pipe
//
// Each test streams planes of 16 floats from an L1-resident buffer into X and
// issues one fma32 per plane on a rotating set of Z tiles, like the kernel's
// s-loop.  "dist" is how many planes ahead the load runs (X registers used as
// a ring of 8): dist = 0 loads the plane and uses it immediately, dist = 4
// keeps four loads in flight.  Reported in fma32 per ns (1 per cycle at the
// clock is the resident-operand peak from bw_test's amx_peak) and GFLOP/s.
// Variants: quad loads; stores mixed in at the kernel's ratio (one stz per
// two fma32); and the kernel's actual pattern (quad ldx per 4 fma32 + 2 pair
// stz per 8 fma32).  Single thread: this measures one unit.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include "../amx.h"

using clk = std::chrono::high_resolution_clock;
using namespace amx::op;

static constexpr int PLANES = 512;                 // 32 KB of source planes: L1 resident
static_assert(PLANES % 8 == 0, "the X ring of 8 registers and the quad groups must divide the planes");
alignas(256) static float src[PLANES * 16];
alignas(256) static float dst[64 * 16];
alignas(64)  static float coef[16];

template <typename F>
static double time_best(F&& f, int reps = 5) {
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

static void report(const char* name, double sec, double nfma) {
    std::printf("%-34s %8.3f fma32/ns  %8.1f GFLOP/s\n", name, 1e-9 * nfma / sec, 1e-9 * nfma * 512 / sec);
}

int main() {
    for (int i = 0; i < PLANES * 16; ++i) {
        src[i] = 1.0f + 1e-6f * (i % 97);
    }
    for (int i = 0; i < 16; ++i) {
        coef[i] = 0.5f + 1e-3f * i;
    }
    const std::size_t iters = 4000;                // x PLANES fma32 each
    const double nfma = double(iters) * PLANES;

    AMX_SET();
    AMX_LDY(ldxy(0, coef));

    // resident operands: the ceiling (same as bw_test's amx_peak, one thread)
    report("resident (no loads)", time_best([&] {
        for (std::size_t it = 0; it < iters; ++it) {
            for (int p = 0; p < PLANES; ++p) {
                AMX_FMA32(fma(p & 3, 0, 0, 0));
            }
        }
    }), nfma);

    // single loads, load-to-use distance d
    for (int d : {0, 1, 2, 4, 7}) {
        char name[64];
        std::snprintf(name, sizeof name, "single ldx, dist %d", d);
        report(name, time_best([&] {
            for (std::size_t it = 0; it < iters; ++it) {
                for (int p = 0; p < d; ++p) {
                    AMX_LDX(ldxy(p & 7, src + p * 16));
                }
                for (int p = 0; p < PLANES; ++p) {
                    const int q = p + d;
                    if (q < PLANES) {
                        AMX_LDX(ldxy(q & 7, src + q * 16));
                    }
                    AMX_FMA32(fma(p & 3, 64 * (p & 7), 0, 0));
                }
            }
        }), nfma);
    }

    // quad loads: 4 planes per ldx, X0..3 / X4..7 alternating (dist 0 and 4)
    for (int ahead : {0, 1}) {
        char name[64];
        std::snprintf(name, sizeof name, "quad ldx, %s", ahead ? "one group ahead" : "immediate");
        report(name, time_best([&] {
            for (std::size_t it = 0; it < iters; ++it) {
                if (ahead) {
                    AMX_LDX(ldxy(0, src, false, true));
                }
                for (int g = 0; g < PLANES / 4; ++g) {
                    const int cur = (g & 1) * 4;
                    if (ahead) {
                        if (g + 1 < PLANES / 4) {
                            AMX_LDX(ldxy(cur ^ 4, src + (g + 1) * 64, false, true));
                        }
                    } else {
                        AMX_LDX(ldxy(cur, src + g * 64, false, true));
                    }
                    for (int k = 0; k < 4; ++k) {
                        AMX_FMA32(fma((4 * g + k) & 3, 64 * (cur + k), 0, 0));
                    }
                }
            }
        }), nfma);
    }

    // stores mixed in: one stz per two fma32 (the kernel's ratio after pair stores
    // is one pair stz per four fma32; here single stz per two, equivalent bytes)
    report("single ldx dist 4 + stz per 2 fma", time_best([&] {
        for (std::size_t it = 0; it < iters; ++it) {
            for (int p = 0; p < 4; ++p) {
                AMX_LDX(ldxy(p & 7, src + p * 16));
            }
            for (int p = 0; p < PLANES; ++p) {
                const int q = p + 4;
                if (q < PLANES) {
                    AMX_LDX(ldxy(q & 7, src + q * 16));
                }
                AMX_FMA32(fma(p & 3, 64 * (p & 7), 0, 0));
                if (p & 1) {
                    AMX_STZ(stz((p >> 1) & 63, dst + ((p >> 1) & 63) * 16));
                }
            }
        }
    }), nfma);

    // the kernel's pattern: quad ldx (one group ahead) per 4 fma32, pair stz per 4 fma32
    report("kernel pattern (quad+pair)", time_best([&] {
        for (std::size_t it = 0; it < iters; ++it) {
            AMX_LDX(ldxy(0, src, false, true));
            for (int g = 0; g < PLANES / 4; ++g) {
                const int cur = (g & 1) * 4;
                if (g + 1 < PLANES / 4) {
                    AMX_LDX(ldxy(cur ^ 4, src + (g + 1) * 64, false, true));
                }
                for (int k = 0; k < 4; ++k) {
                    AMX_FMA32(fma((4 * g + k) & 3, 64 * (cur + k), 0, 0));
                }
                AMX_STZ(stz((2 * g) & 62, dst + ((2 * g) & 62) * 16, true));
            }
        }
    }), nfma);

    AMX_CLR();
    std::printf("(1 fma32/ns at 1 GHz; the resident line is the unit's clock in fma32/ns)\n");
    return 0;
}
