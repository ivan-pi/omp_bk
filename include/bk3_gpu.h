// bk3_gpu.h -- sizes and SLM layout of the BK3 GPU kernels, shared by
// openmp/offload/bk3_offload.cpp and sycl/bk3_sycl.cpp.

#ifndef BK3_GPU_H
#define BK3_GPU_H

#include <algorithm>

#include "bk_common.h"

namespace bk3gpu {

using bk::round_up;

// (e, a, b) of a flat index t over an (*, na, nb) box, b fastest
struct idx3 {
    int e;
    int a;
    int b;
};

template <int na, int nb>
constexpr idx3 split3(const int t)
{
    return {t / (na * nb), t % (na * nb) / nb, t % nb};
}

// fluxes from the derivatives at a point, G the six symmetric factors
template <typename T>
inline void apply_g(const T (&g)[6], const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    rr = g[0] * qr + g[1] * qs + g[2] * qt;
    rs = g[1] * qr + g[3] * qs + g[4] * qt;
    rt = g[2] * qr + g[4] * qs + g[5] * qt;
}

// cu: elements per work-group, at most `want`; SLM (B, D and four nq^3
// boxes per element) within 60 KB and one work-item per (q, r) within 1024
constexpr int cu_batch(const int nq, const int want)
{
    const int by_slm = (61440 / 4 - nq * nq * 2) / (4 * nq * nq * nq);
    return std::max(1, std::min({want, by_slm, 1024 / (nq * nq)}));
}

constexpr int cu_block(const int nq, const int batch) { return round_up(batch * nq * nq, 16); }

// sf: (e, a, b, c) of a box at e * se + a * sa + b * sb + c
struct box_strides {
    int sb;
    int sa;
    int se;
};

// rows of odd length (sb = nq, or nq + 1 for even nq), so a sweep along c
// by lanes that differ in b hits distinct banks
constexpr box_strides box_dense(const int nq)
{
    const int sb = (nq % 2 == 0) ? nq + 1 : nq;
    return {sb, nq * sb, nq * nq * sb};
}

// the strides with the fewest SLM bank conflicts over all sf steps for
// SIMD8 messages (scripts/study/bk3_slm_pad.py); at nq = 9 box_dense is
// already conflict-free
constexpr box_strides box_padded(const int nq)
{
    switch (nq) {
        case 3: return {7, 27, 83};
        case 4: return {6, 25, 107};
        case 5: return {5, 25, 128};
        case 6: return {9, 60, 362};
        case 7: return {7, 59, 417};
        case 8: return {9, 78, 626};
        case 10: return {13, 130, 1314};
        default: return box_dense(nq);
    }
}

constexpr box_strides box_layout(const int nq, const bool padded) { return padded ? box_padded(nq) : box_dense(nq); }

// sf: elements per work-group, at most `want`; three boxes within 60 KB and
// one work-item per (q, r) within 1024
constexpr int sf_batch(const int nq, const int want, const bool padded = false)
{
    const int by_slm = 61440 / (3 * box_layout(nq, padded).se * 4);
    return std::max(1, std::min({want, by_slm, 1024 / (nq * nq)}));
}

constexpr int sf_block(const int nq, const int batch) { return std::min(round_up(batch * nq * nq, 16), 1024); }

// B (nm x nq) and D (nq x nq) copied into the caller's local arrays, which
// then stay in registers: uniform values, scalar operands of the FMAs,
// instead of loads in every step
template <typename T, int nq>
inline void load_basis(const T* __restrict__ gB, const T* __restrict__ gD, T (&B)[(nq - 1) * nq], T (&D)[nq * nq])
{
    #pragma unroll
    for (int x = 0; x < (nq - 1) * nq; ++x) {
        B[x] = gB[x];
    }
    #pragma unroll
    for (int x = 0; x < nq * nq; ++x) {
        D[x] = gD[x];
    }
}

} // namespace bk3gpu

#endif // BK3_GPU_H
