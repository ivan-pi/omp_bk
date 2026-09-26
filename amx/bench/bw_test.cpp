// bw_test.cpp -- memory-bandwidth roof for the BK1 study: STREAM-style kernels
// (copy, scale, add, triad), the Schoenauer triad d = a + b*c (3 reads, 1
// write), read-only (read, dot) and write-only (fill) tests, with the same OpenMP setup
// and thread default as BK1_amx.cpp, so the numbers are directly comparable.
// On Apple Silicon the read, fill and copy streams are repeated through the
// AMX unit's own loads and stores (amx_read, amx_fill, amx_copy), which is how
// the elements-on-lanes kernel reads `in` and writes `out`.
//
//   make bw_test       (or: clang++ -O3 -mcpu=native -std=c++17 -fopenmp amx/bench/bw_test.cpp -o bw_test)
//   ./bw_test [floats_per_array=67108864] [ntests=5]  > bw.csv
//
// Output: CSV lines "test,threads,bytes,seconds,GB_per_s", best of ntests.
// Bytes counted are the compulsory bytes each test moves (no write-allocate
// term); the "fill" test shows what a pure write stream achieves and so
// whether write-allocate is visible on this machine.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../amx.h"     // for the AMX peak test (real hardware only)

using clk = std::chrono::high_resolution_clock;

// 4-float vector for the reductions and the FMA-peak loop (GNU vector types)
typedef float v4f __attribute__((vector_size(16)));
static inline v4f v4_load(const float* p) {
    v4f v;
    std::memcpy(&v, p, 16);
    return v;
}
static inline float v4_sum(v4f v) { return v[0] + v[1] + v[2] + v[3]; }

template <typename F>
static double best_of(int ntests, F&& f) {
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < ntests; ++t) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (std::size_t(1) << 26);
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 5;
    assert(n >= 32 && ntests >= 1);          // the reductions walk 32 floats per step
    int threads = 1;
#ifdef _OPENMP
    if (!std::getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() - 1));
    }
    threads = omp_get_max_threads();
#endif
    std::vector<float> a(n), b(n), c(n), d(n);
    // first touch inside a parallel region, as the benchmark arrays are
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
        d[i] = 0.0f;
    }

    const double bytes = double(n) * sizeof(float);
    volatile float sink = 0.f;
    const float scalar = 3.0f;

    auto report = [&](const char* name, double nbytes, double s) {
        std::printf("%s,%d,%.0f,%.6f,%.2f\n", name, threads, nbytes, s, 1e-9 * nbytes / s);
    };
    // Every array a test writes is read afterwards into the volatile sink, so
    // no store stream can be judged dead (outside the timed region; a few
    // strided elements are enough to keep the writes observable).
    auto consume = [&](const std::vector<float>& v) {
        float acc = 0.f;
        for (std::size_t i = 0; i < n; i += n / 64 + 1) {
            acc += v[i];
        }
        sink = sink + acc;
    };

    // copy:  c = a                      2 arrays
    report("copy", 2 * bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i];
        }
    }));
    consume(c);
    // scale: b = s * c                  2 arrays
    report("scale", 2 * bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = scalar * c[i];
        }
    }));
    consume(b);
    // add:   c = a + b                  3 arrays
    report("add", 3 * bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }));
    consume(c);
    // triad: a = b + s * c              3 arrays
    report("triad", 3 * bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
    }));
    consume(a);
    // striad: d = a + b * c  (Schoenauer triad)   4 arrays: 3 reads, 1 write
    report("striad", 4 * bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = a[i] + b[i] * c[i];
        }
    }));
    consume(d);
    // dot:   sum a[i]*b[i]              2 arrays, read only.  Eight independent
    // vector accumulators per thread so the loop is bandwidth-bound, not
    // latency-bound (a single scalar reduction chain caps at ~4 GB/s/thread).
    report("dot", 2 * bytes, best_of(ntests, [&] {
        double total = 0.0;
        #pragma omp parallel reduction(+ : total)
        {
            v4f acc[8] = {};
            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < n - 31; i += 32) {
                for (int k = 0; k < 8; ++k) {
                    acc[k] += v4_load(&a[i + 4 * k]) * v4_load(&b[i + 4 * k]);
                }
            }
            for (int k = 0; k < 8; ++k) {
                total += v4_sum(acc[k]);
            }
        }
        sink = float(total);
    }));
    // read:  sum a[i]                   1 array, read only
    report("read", bytes, best_of(ntests, [&] {
        double total = 0.0;
        #pragma omp parallel reduction(+ : total)
        {
            v4f acc[8] = {};
            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < n - 31; i += 32) {
                for (int k = 0; k < 8; ++k) {
                    acc[k] += v4_load(&a[i + 4 * k]);
                }
            }
            for (int k = 0; k < 8; ++k) {
                total += v4_sum(acc[k]);
            }
        }
        sink = float(total);
    }));
    // fill:  c = s                      1 array, write only
    report("fill", bytes, best_of(ntests, [&] {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = scalar;
        }
    }));
    consume(c);
    // fma_peak: in-register FMA throughput per thread, summed over threads --
    // the compute roof at whatever clock the machine actually runs.
    // On arm64 the loop is inline assembly: exactly sixteen independent
    // fmla.4s per iteration (16 accumulator registers, 4 pipes x ~4-cycle
    // latency), so the compiler cannot add anything to the instruction stream.
    // Elsewhere a C loop with sixteen named v4f accumulators is used.
    {
        const std::size_t iters = std::size_t(1) << 24;
        double best = std::numeric_limits<double>::max();
        double flops = 0.0;
        for (int t = 0; t < ntests; ++t) {
            auto t0 = clk::now();
            double f = 0.0;
            #pragma omp parallel reduction(+ : f)
            {
#if defined(__aarch64__)
                std::size_t n = iters;
                float out;
                __asm__ volatile(
                    "movi v0.4s, #0\n movi v1.4s, #0\n movi v2.4s, #0\n movi v3.4s, #0\n"
                    "movi v4.4s, #0\n movi v5.4s, #0\n movi v6.4s, #0\n movi v7.4s, #0\n"
                    "movi v8.4s, #0\n movi v9.4s, #0\n movi v10.4s, #0\n movi v11.4s, #0\n"
                    "movi v12.4s, #0\n movi v13.4s, #0\n movi v14.4s, #0\n movi v15.4s, #0\n"
                    "fmov v16.4s, #1.0\n fmov v17.4s, #0.5\n"
                    "1:\n"
                    "fmla v0.4s, v16.4s, v17.4s\n fmla v1.4s, v16.4s, v17.4s\n"
                    "fmla v2.4s, v16.4s, v17.4s\n fmla v3.4s, v16.4s, v17.4s\n"
                    "fmla v4.4s, v16.4s, v17.4s\n fmla v5.4s, v16.4s, v17.4s\n"
                    "fmla v6.4s, v16.4s, v17.4s\n fmla v7.4s, v16.4s, v17.4s\n"
                    "fmla v8.4s, v16.4s, v17.4s\n fmla v9.4s, v16.4s, v17.4s\n"
                    "fmla v10.4s, v16.4s, v17.4s\n fmla v11.4s, v16.4s, v17.4s\n"
                    "fmla v12.4s, v16.4s, v17.4s\n fmla v13.4s, v16.4s, v17.4s\n"
                    "fmla v14.4s, v16.4s, v17.4s\n fmla v15.4s, v16.4s, v17.4s\n"
                    "subs %0, %0, #1\n"
                    "b.ne 1b\n"
                    "fadd v0.4s, v0.4s, v8.4s\n"
                    "fmov %w1, s0\n"
                    : "+r"(n), "=r"(out)
                    :
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                      "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
                sink = out;
#else
                const v4f x = {1.000001f, 0.999999f, 1.000002f, 0.999998f};
                const v4f y = {1e-7f, -1e-7f, 2e-7f, -2e-7f};
                v4f a0 = {1, 2, 3, 4};
                v4f a1 = a0 * 2.f;
                v4f a2 = a0 * 3.f;
                v4f a3 = a0 * 4.f;
                v4f a4 = a0 * 5.f;
                v4f a5 = a0 * 6.f;
                v4f a6 = a0 * 7.f;
                v4f a7 = a0 * 8.f;
                v4f a8 = a0 * 9.f;
                v4f a9 = a0 * 10.f;
                v4f a10 = a0 * 11.f;
                v4f a11 = a0 * 12.f;
                v4f a12 = a0 * 13.f;
                v4f a13 = a0 * 14.f;
                v4f a14 = a0 * 15.f;
                v4f a15 = a0 * 16.f;
                for (std::size_t i = 0; i < iters; ++i) {
                    a0 = a0 * x + y;
                    a1 = a1 * x + y;
                    a2 = a2 * x + y;
                    a3 = a3 * x + y;
                    a4 = a4 * x + y;
                    a5 = a5 * x + y;
                    a6 = a6 * x + y;
                    a7 = a7 * x + y;
                    a8 = a8 * x + y;
                    a9 = a9 * x + y;
                    a10 = a10 * x + y;
                    a11 = a11 * x + y;
                    a12 = a12 * x + y;
                    a13 = a13 * x + y;
                    a14 = a14 * x + y;
                    a15 = a15 * x + y;
                }
                const v4f s4 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
                sink = v4_sum(s4);
#endif
                f += double(iters) * 16 * 4 * 2;          // flops this thread
            }
            auto t1 = clk::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            if (sec < best) {
                best = sec;
                flops = f;
            }
        }
        std::printf("fma_peak,%d,%.0f,%.6f,%.2f\n", threads, flops, best, 1e-9 * flops / best);
    }

#if AMX_HW
    // amx_peak: back-to-back fma32 outer products on the four Z tiles with the
    // operands resident in X/Y (no loads or stores): the unit's FMA-pipe roof.
    // Each fma32 is a 16x16 outer product = 512 flop.  Summed over threads,
    // so with several threads per cluster it measures the shared unit(s).
    {
        const std::size_t iters = std::size_t(1) << 22;
        double best = std::numeric_limits<double>::max();
        double flops = 0.0;
        alignas(64) float xv[16], yv[16];
        for (int k = 0; k < 16; ++k) {
            xv[k] = 1.0f + 1e-6f * k;
            yv[k] = 1e-7f * (k + 1);
        }
        for (int t = 0; t < ntests; ++t) {
            auto t0 = clk::now();
            double f = 0.0;
            #pragma omp parallel reduction(+ : f)
            {
                AMX_SET();
                AMX_LDX(amx::op::ldxy(0, xv));
                AMX_LDY(amx::op::ldxy(0, yv));
                for (int tile = 0; tile < 4; ++tile) { AMX_FMA32(amx::op::fma(tile, 0, 0, 1)); }   // z = x*y
                for (std::size_t i = 0; i < iters; ++i) {
                    AMX_FMA32(amx::op::fma(0, 0, 0, 0));
                    AMX_FMA32(amx::op::fma(1, 0, 0, 0));
                    AMX_FMA32(amx::op::fma(2, 0, 0, 0));
                    AMX_FMA32(amx::op::fma(3, 0, 0, 0));
                }
                alignas(64) float row[16];
                AMX_STZ(amx::op::stz(0, row));
                AMX_CLR();
                sink = row[0];
                f += double(iters) * 4 * 512;
            }
            auto t1 = clk::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            if (sec < best) {
                best = sec;
                flops = f;
            }
        }
        std::printf("amx_peak,%d,%.0f,%.6f,%.2f\n", threads, flops, best, 1e-9 * flops / best);
    }

    // amx_read / amx_fill / amx_copy: the read, fill and copy streams issued
    // by the AMX unit instead of the core, 64-byte lines through a ring of
    // X registers (loads) or Z rows (stores), threads as above.  Read against
    // read / fill / copy: the gap is what the kernel's DRAM-facing steps lose
    // (ldx gets no help from the core's prefetchers; a single stz writes half
    // of a 128-byte line).  amx_copy is the kernel's own step shape, one
    // fma32 per line between the load and the store (Y0 = e0 so Z row 0 of
    // the tile is a copy of X).
    {
        // whole 64-byte lines from the first aligned line of each array
        auto first_line = [](const float* p) {
            return (64 - reinterpret_cast<uintptr_t>(p) % 64) % 64 / sizeof(float);
        };
        const std::size_t oa = first_line(a.data());
        const std::size_t oc = first_line(c.data());
        const std::size_t nl = std::min(n - oa, n - oc) / 16;         // lines per stream
        assert(nl >= 64);
        const float* ra = a.data() + oa;
        float* wc = c.data() + oc;
        const double line_bytes = double(nl) * 64;
        alignas(64) float e0[16] = {1.0f};                              // y = (1, 0, ..., 0)

        report("amx_read", line_bytes, best_of(ntests, [&] {
            #pragma omp parallel
            {
                AMX_SET();
                #pragma omp for schedule(static)
                for (std::size_t l = 0; l < nl; ++l) {
                    AMX_LDX(amx::op::ldxy(int(l & 7), ra + 16 * l));
                }
                AMX_CLR();
            }
        }));
        report("amx_fill", line_bytes, best_of(ntests, [&] {
            #pragma omp parallel
            {
                AMX_SET();
                AMX_LDX(amx::op::ldxy(0, ra));                          // every Z row = x[0] * (1, 0, ..)
                AMX_LDY(amx::op::ldxy(0, e0));
                for (int tile = 0; tile < 4; ++tile) {
                    AMX_FMA32(amx::op::fma(tile, 0, 0, amx::op::ALU_MUL));
                }
                #pragma omp for schedule(static)
                for (std::size_t l = 0; l < nl; ++l) {
                    AMX_STZ(amx::op::stz(int(l & 63), wc + 16 * l));
                }
                AMX_CLR();
            }
        }));
        consume(c);
        report("amx_copy", 2 * line_bytes, best_of(ntests, [&] {
            #pragma omp parallel
            {
                AMX_SET();
                AMX_LDY(amx::op::ldxy(0, e0));
                #pragma omp for schedule(static)
                for (std::size_t l = 0; l < nl; ++l) {
                    const int x = int(l & 7);
                    const int tile = int(l & 3);
                    AMX_LDX(amx::op::ldxy(x, ra + 16 * l));
                    AMX_FMA32(amx::op::fma(tile, 64 * x, 0, amx::op::ALU_MUL));
                    AMX_STZ(amx::op::stz(amx::op::zrow_f32(tile, 0), wc + 16 * l));
                }
                AMX_CLR();
            }
        }));
        for (std::size_t l = 0; l < nl; l += nl / 64 + 1) {            // it is a copy
            assert(std::memcmp(wc + 16 * l, ra + 16 * l, 64) == 0);
        }
        consume(c);
    }
#endif
    // "add" (2 reads, 1 write) is the BK1-shaped mix and the number to use for
    // the memory roof; "striad" (3 reads, 1 write) is the closest to BK3/BK5;
    // "fma_peak" and "amx_peak" are the compute roofs (GFLOP/s in column 5).
    return sink == 12345.f;   // keep the reductions alive
}
