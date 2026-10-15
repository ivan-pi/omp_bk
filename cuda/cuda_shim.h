#ifndef CUDA_SHIM_H
#define CUDA_SHIM_H

// cuda_shim.h -- a stand-in for the CUDA keywords, so that a host compiler
// can build and run a __global__ kernel as an ordinary function.  The
// harness sets one thread per block and one block, so every loop of the
// form `for (tid = threadIdx.x; ...; tid += blockDim.x)` covers all its
// points in order and the empty __syncthreads() is exact.  This checks the
// indexing, the phases and the batching of a kernel, not its execution by
// many threads.  float2 and make_float2 stand in for the CUDA vector type.
// Include it before the kernel header, with __CUDACC__ defined on the
// command line so that the kernel header is not empty.

#include <cmath>

#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline

// the vector type of the float2 kernel
struct float2 {
    float x;
    float y;
};

inline float2 make_float2(const float x, const float y) { return float2{x, y}; }

struct cuda_shim_dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

static cuda_shim_dim3 threadIdx{0, 0, 0};
static cuda_shim_dim3 blockIdx{0, 0, 0};
static cuda_shim_dim3 blockDim{1, 1, 1};
static cuda_shim_dim3 gridDim{1, 1, 1};

inline void __syncthreads() {}

#endif // CUDA_SHIM_H
