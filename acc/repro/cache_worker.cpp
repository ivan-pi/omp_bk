// cache_worker.cpp -- standalone reproducer: an OpenACC `cache` directive at
// the top of a gang loop whose body is a `loop worker` over worker-private
// arrays, with `loop vector` phases inside.
//
// Observed with nvc++ 25.3 (NVHPC, linuxarm64) on GH200: with the cache
// directive this kernel faults at n = 4 (CUDA_ERROR_ILLEGAL_INSTRUCTION,
// 16 workers of 64 lanes) and is correct at the other n; without the
// directive (-DNO_CACHE) it is correct at every n. The full kernel it was
// reduced from, the BK1 worker kernel of acc/BK1_acc.cpp in this
// repository, shows the same fault at nq = 4 and additionally
// CUDA_ERROR_MISALIGNED_ADDRESS at nq = 7..10; this file does not reproduce
// the second symptom. The same directive at the top of a gang loop without
// a worker loop below it works in every kernel of acc/BK1_acc.cpp.
//
// build:  nvc++ -O3 -std=c++17 -acc=gpu -Minfo=acc -o cache_worker cache_worker.cpp
//         nvc++ -O3 -std=c++17 -acc=gpu -DNO_CACHE -o cache_worker_nocache cache_worker.cpp
// run:    ./cache_worker <n> [nelmt]         n = 3..10 (box edge; nq of BK1)
//         ./run.sh                           both builds, every n, one process each
//
// Shape of the kernel: a gang handles W elements, one per worker; each
// element has a box of n^3 values in a worker-private array; phase 1 fills
// the box from global memory scaled by the cached table, phase 2 writes out
// a neighbour of each box entry, so the barrier between the phases matters.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(NO_CACHE)
#define CACHE_TABLE
#elif defined(__NVCOMPILER)
#define CACHE_TABLE _Pragma("acc cache(readonly: tab[0:ntab])")
#else
#define CACHE_TABLE _Pragma("acc cache(tab[0:ntab])")   /* GCC: no readonly modifier */
#endif

constexpr int round_to_warps(const int lanes)
{
    const int rounded = ((lanes + 31) / 32) * 32;
    return rounded > 512 ? 512 : rounded;
}

template <int n>
int run(const int nelmt)
{
    constexpr int box  = n * n * n;
    constexpr int ntab = (n - 1) * n;          // the BK1 basis: nm x nq entries
    constexpr int V    = round_to_warps(box);  // lanes per worker
    constexpr int W    = 1024 / V;             // workers (elements) per gang

    std::vector<float> table(ntab);
    std::vector<float> in(std::size_t(nelmt) * box);
    std::vector<float> out(std::size_t(nelmt) * box, 0.0f);
    for (int i = 0; i < ntab; ++i) {
        table[i] = std::cos(float(i));
    }
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = float(i % 97) - 48.0f;
    }

    const float* tab = table.data();
    const float* x   = in.data();
    float*       y   = out.data();
    const int ngang  = (nelmt + W - 1) / W;

    #pragma acc parallel loop gang num_workers(W) vector_length(V) \
        copyin(tab[0:ntab], x[0:nelmt*box]) copyout(y[0:nelmt*box])
    for (int g = 0; g < ngang; ++g) {
        CACHE_TABLE
        #pragma acc loop worker
        for (int w = 0; w < W; ++w) {
            const int e0 = g * W + w;
            const int e  = e0 < nelmt ? e0 : nelmt - 1;   // tail: recompute the last element
            float s[box];                                  // worker-private box

            #pragma acc loop vector
            for (int i = 0; i < box; ++i) {
                s[i] = x[e * box + i] * tab[i % ntab];
            }
            #pragma acc loop vector
            for (int i = 0; i < box; ++i) {
                if (e0 < nelmt) {
                    y[e * box + i] = s[(i + 1) % box] + tab[i % ntab];
                }
            }
        }
    }

    // host reference
    double maxdiff = 0.0;
    for (int e = 0; e < nelmt; ++e) {
        for (int i = 0; i < box; ++i) {
            const int j = (i + 1) % box;
            const double ref = double(in[std::size_t(e) * box + j]) * table[j % ntab] + table[i % ntab];
            const double d = std::fabs(ref - out[std::size_t(e) * box + i]);
            if (d > maxdiff) {
                maxdiff = d;
            }
        }
    }
    std::printf("n = %2d  W = %2d  V = %3d  nelmt = %d  max|diff| = %.3g  %s\n",
                n, W, V, nelmt, maxdiff, maxdiff < 1e-4 ? "ok" : "MISMATCH");
    return maxdiff < 1e-4 ? 0 : 1;
}

int main(int argc, char** argv)
{
    const int n     = argc > 1 ? std::atoi(argv[1]) : 7;
    const int nelmt = argc > 2 ? std::atoi(argv[2]) : 1001;   // not a multiple of any W
    switch (n) {
        case 3:  return run<3>(nelmt);
        case 4:  return run<4>(nelmt);
        case 5:  return run<5>(nelmt);
        case 6:  return run<6>(nelmt);
        case 7:  return run<7>(nelmt);
        case 8:  return run<8>(nelmt);
        case 9:  return run<9>(nelmt);
        case 10: return run<10>(nelmt);
        default:
            std::fprintf(stderr, "n must be 3..10\n");
            return 2;
    }
}
