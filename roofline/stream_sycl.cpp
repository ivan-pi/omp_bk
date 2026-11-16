// stream_sycl.cpp -- the STREAM-like bandwidth test of stream_omp.cpp in SYCL,
// so a SYCL-only build tree has its own roofs: the same kernels on the same
// device-allocated (USM) float arrays, the same byte counts and the same
// output format.  Each kernel runs ntests times after a warm-up; the minimum
// host time (submit to completion) is reported, as stream_omp reports its
// minimum wall time.
//
//   ./stream_sycl [n = 2^28] [ntests = 20]
//
// Kernels: read (sum reduction, 1 read), write (fill, 1 write), copy (1R+1W),
// scale (1R+1W), add (2R+1W), triad (2R+1W), bk1p2 (BK1's traffic at p = 2,
// one element per work-item) and read9 (BK3's mix, 9 reads per write; 256
// consecutive work-items read one contiguous block).  The last line checks
// the read kernel's sum.

#include "bk_sycl.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace {

void report(const char* name, const double bytes, const double seconds)
{
    std::printf("%-6s bytes = %10.3e  time = %9.3f ms  GB/s = %8.2f\n",
                name, bytes, 1e3 * seconds, 1e-9 * bytes / seconds);
}

// floats read9 writes: blocks of 256, nine reads each
std::size_t read9_written(const std::size_t n) { return n / (9 * 256) * 256; }

} // namespace

int main(int argc, char** argv)
{
    const std::size_t n = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 28;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 20;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    float* a = sycl::malloc_device<float>(n, q);
    float* b = sycl::malloc_device<float>(n, q);
    float* c = sycl::malloc_device<float>(n, q);
    float* sum = sycl::malloc_shared<float>(1, q);
    if (a == nullptr || b == nullptr || c == nullptr || sum == nullptr) {
        std::fprintf(stderr, "sycl::malloc_device failed for n = %zu\n", n);
        return 1;
    }
    std::printf("device %s, n = %zu floats (%.2f GB per array)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                n, 1e-9 * n * sizeof(float));

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }).wait();

    const float s = 3.0f;
    auto read = [&] {
        *sum = 0.0f;
        return q.parallel_for(sycl::range<1>(n), sycl::reduction(sum, sycl::plus<float>()),
                              [=](sycl::id<1> i, auto& acc) { acc += a[i]; });
    };
    auto write = [&] {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { c[i] = 0.5f; });
    };
    auto copy = [&] {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { c[i] = a[i]; });
    };
    auto scale = [&] {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = s * c[i]; });
    };
    auto add = [&] {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { c[i] = a[i] + b[i]; });
    };
    auto triad = [&] {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = b[i] + s * c[i]; });
    };
    // BK1 traffic at p = 2 (nq = 4, nm = 3): per element 27 values of in (a)
    // and 64 of JxW (b) read, 27 of out (c) written.
    const std::size_t ne = n / 64;
    auto bk1p2 = [&] {
        return q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> id) {
            const std::size_t e = id[0];
            float w = 0.0f;
            for (int k = 0; k < 64; ++k) {
                w += b[e * 64 + k];
            }
            for (int k = 0; k < 27; ++k) {
                c[e * 27 + k] = w * a[e * 27 + k];
            }
        });
    };
    const std::size_t nw = read9_written(n);
    auto read9 = [&] {
        return q.parallel_for(sycl::range<1>(nw), [=](sycl::id<1> id) {
            const std::size_t i = id[0];
            const std::size_t blk = i / 256;
            const std::size_t l = i % 256;
            float t = 0.0f;
            for (int j = 0; j < 9; ++j) {
                t += a[(blk * 9 + j) * 256 + l];
            }
            c[i] = t;
        });
    };

    const double fb = sizeof(float);
    report("read",  fb * n,     bk::time_launches(ntests, read, false).host);
    const float read_sum = *sum;
    report("write", fb * n,     bk::time_launches(ntests, write, false).host);
    report("copy",  2 * fb * n, bk::time_launches(ntests, copy, false).host);
    report("scale", 2 * fb * n, bk::time_launches(ntests, scale, false).host);
    report("add",   3 * fb * n, bk::time_launches(ntests, add, false).host);
    report("triad", 3 * fb * n, bk::time_launches(ntests, triad, false).host);
    report("bk1p2", fb * ne * (27 + 64 + 27), bk::time_launches(ntests, bk1p2, false).host);
    report("read9", fb * nw * 10, bk::time_launches(ntests, read9, false).host);

    // after copy, scale, add and the triads: a = b + 3 c with b = 3 c, c = a + b
    read().wait();
    std::printf("check: sum(a) before = %.6e (expect %.6e), sum(a) after = %.6e (nonzero)\n",
                double(read_sum), double(n), double(*sum));
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    sycl::free(sum, q);
    return 0;
}
