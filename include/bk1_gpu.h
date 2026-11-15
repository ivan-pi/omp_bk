// bk1_gpu.h -- helpers shared by the BK1 GPU kernels: the OpenMP offload
// variants (openmp/offload/bk1_offload.cpp) and SYCL (sycl/bk1_sycl.cpp).

#ifndef BK1_GPU_H
#define BK1_GPU_H

namespace bk1gpu {

constexpr int round_up(const int n, const int m) { return ((n + m - 1) / m) * m; }

// Threads per team: one per point of the nq^3 box, whole sub-groups of 16.
#ifndef TEAM_ROUND
#define TEAM_ROUND 16
#endif

#define UNROLL _Pragma("unroll")

// elements per team of the plane kernels: fill about `threads` threads
constexpr int plane_batch(const int nq, const int threads)
{
    return (threads / (nq * nq) > 1) ? threads / (nq * nq) : 1;
}

// the largest batch up to `want` elements per team that fits a 1024-thread
// work-group (one thread per point of an nq x nq plane per element) and 60 KB
// of the 64 KB of SLM per Xe-core (two nq^3 float boxes per element)
constexpr int cap_batch(const int nq, const int want)
{
    const int by_threads = 1024 / (nq * nq);
    const int by_slm = 61440 / (2 * nq * nq * nq * 4);
    int b = want;
    if (b > by_threads) {
        b = by_threads;
    }
    if (b > by_slm) {
        b = by_slm;
    }
    return (b > 1) ? b : 1;
}

// threads per team for the grid-strided kernels: the widest step (batch *
// nq * nq work items), rounded up to whole sub-groups of 16
constexpr int gs_block(const int nq, const int batch)
{
    return ((batch * nq * nq + 15) / 16) * 16;
}

// SLM strides of the seven-step kernel's five intermediate boxes, each
// (element, a, b, c) with c contiguous: {element stride, a stride, b stride}.
//   A s2->s3 (e,p,j,k)  B s3->s4 (e,q,p,k)  C s4->s6 (e,r,q,p)
//   D s6->s7 (e,k,q,p)  E s7->s8 (e,k,j,p)
// dense: no padding.  padded: chosen per order by scripts/study/slm_pad_search.py
// to spread the lanes of a SIMD16 sub-group over the 16 SLM banks (at the
// order's best batch; valid for any batch).
struct box_strides {
    int e;
    int a;
    int b;
};
struct slm_layout {
    box_strides A, B, C, D, E;
};

constexpr slm_layout slm_dense(const int nq)
{
    const int nm = nq - 1;
    return {{nq * nm * nm, nm * nm, nm}, {nq * nq * nm, nq * nm, nm}, {nq * nq * nq, nq * nq, nq},
            {nm * nq * nq, nq * nq, nq}, {nm * nm * nq, nm * nq, nq}};
}

constexpr slm_layout slm_padded(const int nq)
{
    switch (nq) {
        case 3: return {{12, 4, 2}, {22, 7, 2}, {27, 9, 3}, {21, 10, 3}, {12, 6, 3}};
        case 4: return {{41, 10, 3}, {60, 15, 3}, {64, 16, 4}, {60, 20, 5}, {39, 13, 4}};
        case 5: return {{90, 18, 4}, {125, 25, 5}, {137, 27, 5}, {105, 26, 5}, {80, 20, 5}};
        case 6: return {{150, 25, 5}, {180, 30, 5}, {216, 36, 6}, {190, 38, 6}, {150, 30, 6}};
        case 7: return {{266, 38, 6}, {294, 42, 6}, {343, 49, 7}, {312, 52, 7}, {252, 42, 7}};
        case 8: return {{416, 52, 7}, {448, 56, 7}, {512, 64, 8}, {504, 72, 9}, {441, 63, 9}};
        case 9: return {{576, 64, 8}, {729, 81, 9}, {729, 81, 9}, {648, 81, 9}, {592, 74, 9}};
        case 10: return {{833, 83, 9}, {922, 92, 9}, {1012, 101, 10}, {1098, 122, 12}, {810, 90, 10}};
        default: return slm_dense(nq);
    }
}

constexpr int imax(const int a, const int b) { return (a > b) ? a : b; }

// floats per element of the two SLM buffers: w0 holds the dense input copy
// and boxes B, D; w1 holds A, C, E
constexpr int slm_w0(const int nq, const slm_layout& L)
{
    return imax((nq - 1) * (nq - 1) * (nq - 1), imax(L.B.e, L.D.e));
}
constexpr int slm_w1(const slm_layout& L) { return imax(L.A.e, imax(L.C.e, L.E.e)); }

// "fill" teams: about `threads` work-items of elements, the team exactly
// `threads` wide (a power of two, so its hardware threads divide an
// Xe-core's 128 at SIMD16 or SIMD32)
constexpr int fill_batch(const int nq, const int threads)
{
    return (threads / (nq * nq) > 1) ? threads / (nq * nq) : 1;
}

} // namespace bk1gpu

#endif // BK1_GPU_H
