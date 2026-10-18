// smem_limit.cpp -- probe: which gang-private arrays does nvc++ place in CUDA
// shared memory? A gang loop declares NARR arrays of NFLOAT floats each and
// uses them from two vector loops; -Minfo=acc reports the placement as
// "CUDA shared memory used for a0,..." or stays silent (global memory).
//
// build: nvc++ -O3 -std=c++17 -acc=gpu -Minfo=acc -DNARR=2 -DNFLOAT=512 -c smem_limit.cpp
// run:   ./smem_limit.sh          (the grid of NARR and NFLOAT, one line each)
//
// Background: nvc++ 25.3 placed the two nq^3 boxes of acc/BK1_acc.cpp's gang
// kernel in shared memory up to nq = 7 (1372 bytes each) and not from nq = 8
// (2048 bytes each), and the batched kernel's boxes only at one element per
// gang; there is no diagnostic and no -gpu option for it.

#include <cstdio>
#include <vector>

#ifndef NARR
#define NARR 2
#endif
#ifndef NFLOAT
#define NFLOAT 512
#endif

int main()
{
    const int ngang = 1000;
    std::vector<float> out(std::size_t(ngang) * 32);
    float* y = out.data();

    #pragma acc parallel loop gang vector_length(128) copyout(y[0:ngang*32])
    for (int g = 0; g < ngang; ++g) {
        float a0[NFLOAT];
#if NARR > 1
        float a1[NFLOAT];
#endif
#if NARR > 2
        float a2[NFLOAT];
#endif
#if NARR > 3
        float a3[NFLOAT];
#endif
        #pragma acc loop vector
        for (int i = 0; i < NFLOAT; ++i) {
            a0[i] = float(g + i);
#if NARR > 1
            a1[i] = 2.0f * a0[i];
#endif
#if NARR > 2
            a2[i] = 3.0f * a0[i];
#endif
#if NARR > 3
            a3[i] = 4.0f * a0[i];
#endif
        }
        #pragma acc loop vector
        for (int i = 0; i < 32; ++i) {
            const int j = (i * 7 + 1) % NFLOAT;
            float s = a0[j];
#if NARR > 1
            s += a1[j];
#endif
#if NARR > 2
            s += a2[j];
#endif
#if NARR > 3
            s += a3[j];
#endif
            y[g * 32 + i] = s;
        }
    }
    std::printf("%g\n", out[out.size() - 1]);
    return 0;
}
