// stream_omp.cpp -- STREAM-like bandwidth test for OpenMP target offload.
// float arrays allocated on the device (omp_target_alloc), so no host
// transfers are timed; each kernel runs ntests times, the minimum wall time
// is reported.  Bytes counted as in STREAM (no write-allocate traffic).
//
//   ./stream_omp [n = 2^28] [ntests = 20]
//
// Kernels: read (sum reduction, 1 read), write (fill, 1 write), copy (1R+1W),
// scale (1R+1W), add (2R+1W), triad (2R+1W) and bk1p2 (BK1's traffic at
// p = 2 with one element per thread: 27 values of in and 64 of JxW read,
// 27 of out written per element, each thread on its own contiguous block).
// The last line checks the read kernel's sum.  Build with `make stream_omp`
// (host fallback) or gpu/scripts/build_aot.sh stream_omp (Intel GPU).

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

// Each kernel is a plain function: the device pointers reach the target
// region as by-value parameters (a by-reference lambda capture of the
// pointer is not handled correctly by is_device_ptr).
using kernel_fn = void (*)(float*, float*, float*, std::size_t, float*);

void k_read(float* a, float*, float*, const std::size_t n, float* sum)
{
    float acc = 0.0f;
    #pragma omp target teams distribute parallel for is_device_ptr(a) reduction(+: acc) map(tofrom: acc)
    for (std::size_t i = 0; i < n; ++i) {
        acc += a[i];
    }
    *sum = acc;
}

void k_write(float*, float*, float* c, const std::size_t n, float*)
{
    #pragma omp target teams distribute parallel for is_device_ptr(c)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = 0.5f;
    }
}

void k_copy(float* a, float*, float* c, const std::size_t n, float*)
{
    #pragma omp target teams distribute parallel for is_device_ptr(a, c)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i];
    }
}

void k_scale(float*, float* b, float* c, const std::size_t n, float*)
{
    const float q = 3.0f;
    #pragma omp target teams distribute parallel for is_device_ptr(b, c)
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = q * c[i];
    }
}

void k_add(float* a, float* b, float* c, const std::size_t n, float*)
{
    #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void k_triad(float* a, float* b, float* c, const std::size_t n, float*)
{
    const float q = 3.0f;
    #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = b[i] + q * c[i];
    }
}

// BK1 traffic at p = 2 (nq = 4, nm = 3), one element per thread as in BK1:
// per element 27 values of in (a) and 64 of JxW (b) read, 27 of out (c) written.
void k_bk1p2(float* a, float* b, float* c, const std::size_t n, float*)
{
    const std::size_t ne = n / 64;
    #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
    for (std::size_t e = 0; e < ne; ++e) {
        float w = 0.0f;
        for (int q = 0; q < 64; ++q) {
            w += b[e * 64 + q];
        }
        for (int k = 0; k < 27; ++k) {
            c[e * 27 + k] = w * a[e * 27 + k];
        }
    }
}

double time_min(const int ntests, const kernel_fn f,
                float* a, float* b, float* c, const std::size_t n, float* sum)
{
    f(a, b, c, n, sum);   // warm-up (first launch, page mapping)
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < ntests; ++t) {
        const double t0 = omp_get_wtime();
        f(a, b, c, n, sum);
        const double t1 = omp_get_wtime();
        best = std::min(best, t1 - t0);
    }
    return best;
}

void report(const char* name, const double bytes, const double seconds)
{
    std::printf("%-6s bytes = %10.3e  time = %9.3f ms  GB/s = %8.2f\n",
                name, bytes, 1e3 * seconds, 1e-9 * bytes / seconds);
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t n = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 28;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 20;
    // Without an offload device (a host-only build, or no GPU found) the
    // arrays live on the host and the target regions run as host fallback.
    const bool offload = omp_get_num_devices() > 0;
    const int dev = offload ? omp_get_default_device() : omp_get_initial_device();

    float* a = static_cast<float*>(omp_target_alloc(n * sizeof(float), dev));
    float* b = static_cast<float*>(omp_target_alloc(n * sizeof(float), dev));
    float* c = static_cast<float*>(omp_target_alloc(n * sizeof(float), dev));
    if (a == nullptr || b == nullptr || c == nullptr) {
        std::fprintf(stderr, "omp_target_alloc failed for n = %zu\n", n);
        return 1;
    }
    std::printf("device %d of %d%s, n = %zu floats (%.2f GB per array)\n",
                dev, omp_get_num_devices(), offload ? "" : " (host fallback)",
                n, 1e-9 * n * sizeof(float));

    #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    const double fb = sizeof(float);
    const std::size_t ne = n / 64;
    float sum = 0.0f;
    report("read",  fb * n,     time_min(ntests, k_read,  a, b, c, n, &sum));
    const float read_sum = sum;
    report("write", fb * n,     time_min(ntests, k_write, a, b, c, n, &sum));
    report("copy",  2 * fb * n, time_min(ntests, k_copy,  a, b, c, n, &sum));
    report("scale", 2 * fb * n, time_min(ntests, k_scale, a, b, c, n, &sum));
    report("add",   3 * fb * n, time_min(ntests, k_add,   a, b, c, n, &sum));
    report("triad", 3 * fb * n, time_min(ntests, k_triad, a, b, c, n, &sum));
    report("bk1p2", fb * ne * (27 + 64 + 27), time_min(ntests, k_bk1p2, a, b, c, n, &sum));

    // after copy, scale, add and the triads: a = b + 3 c with b = 3 c, c = a + b
    k_read(a, b, c, n, &sum);
    std::printf("check: sum(a) before = %.6e (expect %.6e), sum(a) after = %.6e (nonzero)\n",
                double(read_sum), double(n), double(sum));
    omp_target_free(a, dev);
    omp_target_free(b, dev);
    omp_target_free(c, dev);
    return 0;
}
