// bk5_offload.cpp -- BK5 (collocated Laplacian at quadrature points) on the GPU:
// the symmetric half of the geometric tensor (6 factors per point, as BK5
// stores it) against the full tensor (9 factors), in two layouts, with the
// original one-element-per-thread kernel and a one-work-item-per-point
// kernel through shared memory.  One order per build (-DNQ=n, nq = p + 1).
//
//   scripts/build.sh bk5_offload_<p>           (-> build/a770/bk5_offload_<p>)
//   scripts/run.sh bk5_offload_<p> [nelmt] [ntests]
//   BK_KERNEL="pt6i_b1 pt9i_b1" ...               run a subset (default: all)
//
// Variant names <kernel><F><layout>[_b<batch>]:
//   ref  the original BK5 kernel (teams loop, one element per thread,
//        3 * nq^3 private work arrays)
//   pt   one work-item per quadrature point, a batch of elements per team:
//        in -> SLM; per point the three derivatives, G applied, three fluxes
//        -> SLM; barrier; transpose derivatives -> out
//   F    6 (symmetric half: rr rs rt ss st tt) or 9 (full 3 x 3, row-major)
//   layout  i: interleaved G(i, j, f, k) per element (BK5's), f: factor-major
//        G(f, i, j, k) per element
//   tid  pt with explicit thread ids and barriers; suffix e: G loaded with
//        in before the first barrier, l: after it (as in pt)
//   sf   sum-factorised sweeps over register columns (as BK1's seven-step
//        kernel), G in layout c: G(f, k, i, j); suffix _b<batch>x<w>: w
//        work-items per column of the batch (1 or 2)
//   cu   port of the CUDA grid-strided kernel (column (e, j, k) per thread
//        walking i, D in SLM, G factor-major); BK_TEAMS=n teams stride over
//        the batches (default: one team per batch)
//   _f<P>  pt with teams of exactly P work-items (512 or 1024) holding as many
//        whole elements as fit (the oneAPI guide's occupancy rule)
// The full tensor is built from the symmetric one, and both apply it in the
// same order, so G6 and G9 results are identical.
//
// Data: pseudo-random in and G (bk::fill_random_test_data), D(i, n) =
// cos(i * nq + n).  Check: largest difference to a double-precision CPU
// reference relative to its largest |value|.  GB/s counts in + out + G
// (F floats per point).

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"

#ifndef NQ
#define NQ 4
#endif

namespace {

constexpr int round16(const int n) { return bk::round_up(n, 16); }

// elements per team of the per-point kernel: at most `want`, with the team
// (one work-item per point) within 1024 and the SLM (4 nq^3 floats per
// element) within 60 KB
constexpr int pt_batch(const int nq, const int want)
{
    const int nq3 = nq * nq * nq;
    int b = want;
    if (b * nq3 > 1024) {
        b = 1024 / nq3;
    }
    if (b * 4 * nq3 * 4 > 61440) {
        b = 61440 / (16 * nq3);
    }
    return (b > 1) ? b : 1;
}

// (e, a, b) of a flat index t over an (*, na, nb) box, b fastest; unpack
// with a structured binding: const auto [e, j, k] = split3<nq, nq>(t);
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

// (e, i, j, k) of a flat point index x over (*, nq, nq, nq), k fastest
struct idx4 {
    int e;
    int i;
    int j;
    int k;
};

template <int nq>
constexpr idx4 split4(const int x)
{
    return {x / (nq * nq * nq), x % (nq * nq * nq) / (nq * nq), x % (nq * nq) / nq, x % nq};
}

// G index of factor f at point (i, j, k) within an element
template <int nq, int F, bool interleaved>
constexpr int gidx(const int i, const int j, const int f, const int k)
{
    if constexpr (interleaved) {
        return ((i * nq + j) * F + f) * nq + k;
    } else {
        return ((f * nq + i) * nq + j) * nq + k;
    }
}

// apply the tensor at a point: (qr, qs, qt) -> (rr, rs, rt)
template <typename T, int nq, int F, bool interleaved>
inline void apply_G(const T* __restrict__ eG, const int i, const int j, const int k,
                    const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    if constexpr (F == 6) {
        const T Grr = eG[gidx<nq, F, interleaved>(i, j, 0, k)];
        const T Grs = eG[gidx<nq, F, interleaved>(i, j, 1, k)];
        const T Grt = eG[gidx<nq, F, interleaved>(i, j, 2, k)];
        const T Gss = eG[gidx<nq, F, interleaved>(i, j, 3, k)];
        const T Gst = eG[gidx<nq, F, interleaved>(i, j, 4, k)];
        const T Gtt = eG[gidx<nq, F, interleaved>(i, j, 5, k)];
        rr = Grr * qr + Grs * qs + Grt * qt;
        rs = Grs * qr + Gss * qs + Gst * qt;
        rt = Grt * qr + Gst * qs + Gtt * qt;
    } else {
        const T G00 = eG[gidx<nq, F, interleaved>(i, j, 0, k)];
        const T G01 = eG[gidx<nq, F, interleaved>(i, j, 1, k)];
        const T G02 = eG[gidx<nq, F, interleaved>(i, j, 2, k)];
        const T G10 = eG[gidx<nq, F, interleaved>(i, j, 3, k)];
        const T G11 = eG[gidx<nq, F, interleaved>(i, j, 4, k)];
        const T G12 = eG[gidx<nq, F, interleaved>(i, j, 5, k)];
        const T G20 = eG[gidx<nq, F, interleaved>(i, j, 6, k)];
        const T G21 = eG[gidx<nq, F, interleaved>(i, j, 7, k)];
        const T G22 = eG[gidx<nq, F, interleaved>(i, j, 8, k)];
        rr = G00 * qr + G01 * qs + G02 * qt;
        rs = G10 * qr + G11 * qs + G12 * qt;
        rt = G20 * qr + G21 * qs + G22 * qt;
    }
}

// ---------------------------------------------------------------------------
// ref: the original BK5 kernel (one element per thread)
// ---------------------------------------------------------------------------
template <typename T, int nq, int F, bool interleaved>
void bk5_ref(const std::size_t nelmt, const T* __restrict__ D, const T* __restrict__ G,
             const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nq3 = nq * nq * nq;
    #pragma omp target teams loop \
        map(to: D[:nq*nq], G[:nelmt*F*nq3], in[:nelmt*nq3]) map(from: out[:nelmt*nq3])
    for (std::size_t e = 0; e < nelmt; ++e) {
        const T* ei = in + e * nq3;
        const T* eG = G + e * (F * nq3);
        T* eo = out + e * nq3;
        T rqr[nq3];
        T rqs[nq3];
        T rqt[nq3];
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    T qr = 0;
                    T qs = 0;
                    T qt = 0;
                    for (int n = 0; n < nq; ++n) {
                        qr += ei[(n * nq + j) * nq + k] * D[i * nq + n];
                    }
                    for (int n = 0; n < nq; ++n) {
                        qs += ei[(i * nq + n) * nq + k] * D[j * nq + n];
                    }
                    for (int n = 0; n < nq; ++n) {
                        qt += ei[(i * nq + j) * nq + n] * D[k * nq + n];
                    }
                    const int x = (i * nq + j) * nq + k;
                    apply_G<T, nq, F, interleaved>(eG, i, j, k, qr, qs, qt, rqr[x], rqs[x], rqt[x]);
                }
            }
        }
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    T tmp = 0;
                    for (int n = 0; n < nq; ++n) {
                        tmp += rqr[(n * nq + j) * nq + k] * D[n * nq + i];
                    }
                    for (int n = 0; n < nq; ++n) {
                        tmp += rqs[(i * nq + n) * nq + k] * D[n * nq + j];
                    }
                    for (int n = 0; n < nq; ++n) {
                        tmp += rqt[(i * nq + j) * nq + n] * D[n * nq + k];
                    }
                    eo[(i * nq + j) * nq + k] = tmp;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pt: one work-item per quadrature point, `batch` elements per team
// ---------------------------------------------------------------------------
template <typename T, int nq, int F, bool interleaved, int batch, int group = 0>
void bk5_pt(const std::size_t nelmt, const T* __restrict__ D, const T* __restrict__ G,
            const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nq3 = nq * nq * nq;
    // group > 0: that many work-items per team (e.g. a power of two, the
    // guide's "fill" rule); otherwise the points rounded up to 16
    constexpr int block = (group > 0) ? group : round16(batch * nq3);
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    static_assert(batch * nq3 <= block, "one work-item per point");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);

    #pragma omp target teams distribute thread_limit(block) \
        map(to: D[:nq*nq], G[:nelmt*F*nq3], in[:nelmt*nq3]) map(from: out[:nelmt*nq3])
    for (int eb = 0; eb < nbatches; ++eb) {
        T u[batch * nq3];
        T rqr[batch * nq3];
        T rqs[batch * nq3];
        T rqt[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nq3);
        const T* b_G = G + std::size_t(eb) * (batch * F * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nq3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int x = 0; x < cb * nq3; ++x) {
                u[x] = b_in[x];
            }
            #pragma omp for
            for (int x = 0; x < cb * nq3; ++x) {
                const auto [e, i, j, k] = split4<nq>(x);
                const T* eu = u + e * nq3;
                T qr = 0;
                T qs = 0;
                T qt = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qr += eu[(n * nq + j) * nq + k] * D[i * nq + n];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qs += eu[(i * nq + n) * nq + k] * D[j * nq + n];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qt += eu[(i * nq + j) * nq + n] * D[k * nq + n];
                }
                apply_G<T, nq, F, interleaved>(b_G + e * (F * nq3), i, j, k, qr, qs, qt, rqr[x], rqs[x], rqt[x]);
            }
            #pragma omp for
            for (int x = 0; x < cb * nq3; ++x) {
                const auto [e, i, j, k] = split4<nq>(x);
                const T* er = rqr + e * nq3;
                const T* es = rqs + e * nq3;
                const T* et = rqt + e * nq3;
                T tmp = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += er[(n * nq + j) * nq + k] * D[n * nq + i];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += es[(i * nq + n) * nq + k] * D[n * nq + j];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += et[(i * nq + j) * nq + n] * D[n * nq + k];
                }
                b_out[x] = tmp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// tid: pt with explicit thread ids and barriers instead of worksharing
// loops, so a work-item keeps its point's data in registers across a
// barrier.  With `early`, the F factors of G are loaded together with `in`,
// before the first barrier, so both global streams are in flight at once;
// otherwise after it, as in pt.  Each team must get `block` threads (one per
// point); the kernel sets short_team if it does not, and the host asserts.
// ---------------------------------------------------------------------------
template <typename T, int nq, int F, bool interleaved>
inline void load_G(const T* __restrict__ eG, const int i, const int j, const int k, T (&g)[F])
{
    #pragma unroll
    for (int f = 0; f < F; ++f) {
        g[f] = eG[gidx<nq, F, interleaved>(i, j, f, k)];
    }
}

template <typename T, int F>
inline void apply_g(const T (&g)[F], const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    if constexpr (F == 6) {
        rr = g[0] * qr + g[1] * qs + g[2] * qt;
        rs = g[1] * qr + g[3] * qs + g[4] * qt;
        rt = g[2] * qr + g[4] * qs + g[5] * qt;
    } else {
        rr = g[0] * qr + g[1] * qs + g[2] * qt;
        rs = g[3] * qr + g[4] * qs + g[5] * qt;
        rt = g[6] * qr + g[7] * qs + g[8] * qt;
    }
}

template <typename T, int nq, int F, bool interleaved, int batch, bool early>
void bk5_tid(const std::size_t nelmt, const T* __restrict__ D, const T* __restrict__ G,
             const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nq3 = nq * nq * nq;
    constexpr int block = round16(batch * nq3);
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);
    int short_team = 0;

    #pragma omp target teams distribute thread_limit(block) \
        map(to: D[:nq*nq], G[:nelmt*F*nq3], in[:nelmt*nq3]) map(from: out[:nelmt*nq3]) \
        map(tofrom: short_team)
    for (int eb = 0; eb < nbatches; ++eb) {
        T u[batch * nq3];
        T rqr[batch * nq3];
        T rqs[batch * nq3];
        T rqt[batch * nq3];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nq3);
        const T* b_G = G + std::size_t(eb) * (batch * F * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nq3);

        #pragma omp parallel num_threads(block)
        {
            const int x = omp_get_thread_num();
            if (x == 0 && omp_get_num_threads() < block) {
                #pragma omp atomic write
                short_team = 1;
            }
            const bool active = x < cb * nq3;
            const auto [e, i, j, k] = split4<nq>(x);
            const T* eG = b_G + e * (F * nq3);
            T g[F];
            if (active) {
                u[x] = b_in[x];
                if constexpr (early) {
                    load_G<T, nq, F, interleaved>(eG, i, j, k, g);
                }
            }
            #pragma omp barrier
            if (active) {
                const T* eu = u + e * nq3;
                T qr = 0;
                T qs = 0;
                T qt = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qr += eu[(n * nq + j) * nq + k] * D[i * nq + n];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qs += eu[(i * nq + n) * nq + k] * D[j * nq + n];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qt += eu[(i * nq + j) * nq + n] * D[k * nq + n];
                }
                if constexpr (!early) {
                    load_G<T, nq, F, interleaved>(eG, i, j, k, g);
                }
                apply_g<T, F>(g, qr, qs, qt, rqr[x], rqs[x], rqt[x]);
            }
            #pragma omp barrier
            if (active) {
                const T* er = rqr + e * nq3;
                const T* es = rqs + e * nq3;
                const T* et = rqt + e * nq3;
                T tmp = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += er[(n * nq + j) * nq + k] * D[n * nq + i];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += es[(i * nq + n) * nq + k] * D[n * nq + j];
                }
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    tmp += et[(i * nq + j) * nq + n] * D[n * nq + k];
                }
                b_out[x] = tmp;
            }
        }
    }
    assert(short_team == 0);
}

// ---------------------------------------------------------------------------
// sf: sum-factorised sweeps over register columns, a batch of elements per
// team, three SLM arrays U, R, S of nq^2 rows of pk floats per element (pk
// = nq, or nq + 1 for even nq so a k-row stride is odd and the lanes of a
// k-column sweep fall in distinct banks).  Per batch:
//   copy   in -> U
//   A | B  thread (e,j,k): R = D_i U along i;  thread (e,i,k): S = D_j U along j
//   C      thread (e,i,j): qt = D_k U along k; per point G applied to
//          (R, S, qt), rr -> R, rs -> S in place, rt kept in registers;
//          U(i,j,:) = D_k^T rt (only this thread reads U(i,j,:) in C)
//   E | F  thread (e,j,k): R = D_i^T R in place; thread (e,i,k): S = D_j^T S
//   out    U + R + S -> out
// The basis index depends only on unrolled loop counters, so D is read
// uniformly.  G in layout c: G(f, k, i, j) per element, so the G loads of
// step C (lanes along (i, j)) are contiguous.
// ---------------------------------------------------------------------------
template <typename T, int nq, int F, int batch, int block>
void bk5_sf(const std::size_t nelmt, const T* __restrict__ D, const T* __restrict__ G,
            const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int pk = (nq % 2 == 0) ? nq + 1 : nq;
    constexpr int ebox = nq2 * pk;
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    static_assert(3 * batch * ebox * int(sizeof(T)) <= 61440, "three SLM arrays within 60 KB");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);

    #pragma omp target teams distribute thread_limit(block) \
        map(to: D[:nq*nq], G[:nelmt*F*nq3], in[:nelmt*nq3]) map(from: out[:nelmt*nq3])
    for (int eb = 0; eb < nbatches; ++eb) {
        T U[batch * ebox];
        T R[batch * ebox];
        T S[batch * ebox];
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nq3);
        const T* b_G = G + std::size_t(eb) * (batch * F * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nq3);

        #pragma omp parallel
        {
            #pragma omp for
            for (int x = 0; x < cb * nq3; ++x) {
                U[x / nq * pk + x % nq] = b_in[x];
            }
            // A | B
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
                T col[nq];
                if (t < cb * nq2) {
                    const auto [e, j, k] = split3<nq, nq>(t);
                    const T* eu = U + e * ebox + j * pk + k;
                    T* er = R + e * ebox + j * pk + k;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        col[n] = eu[n * nq * pk];
                    }
                    #pragma unroll
                    for (int i = 0; i < nq; ++i) {
                        T s = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            s += D[i * nq + n] * col[n];
                        }
                        er[i * nq * pk] = s;
                    }
                } else {
                    const int t2 = t - cb * nq2;
                    const auto [e, i, k] = split3<nq, nq>(t2);
                    const T* eu = U + e * ebox + i * nq * pk + k;
                    T* es = S + e * ebox + i * nq * pk + k;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        col[n] = eu[n * pk];
                    }
                    #pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        T s = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            s += D[j * nq + n] * col[n];
                        }
                        es[j * pk] = s;
                    }
                }
            }
            // C
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                const int e = t / nq2;
                const int c = t % nq2;
                const int base = e * ebox + c * pk;
                const T* eG = b_G + e * (F * nq3) + c;
                T col[nq];
                T rt[nq];
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = U[base + n];
                }
                #pragma unroll
                for (int k = 0; k < nq; ++k) {
                    T qt = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        qt += D[k * nq + n] * col[n];
                    }
                    T g[F];
                    #pragma unroll
                    for (int f = 0; f < F; ++f) {
                        g[f] = eG[(f * nq + k) * nq2];
                    }
                    T rr;
                    T rs;
                    apply_g<T, F>(g, R[base + k], S[base + k], qt, rr, rs, rt[k]);
                    R[base + k] = rr;
                    S[base + k] = rs;
                }
                #pragma unroll
                for (int k = 0; k < nq; ++k) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[n * nq + k] * rt[n];
                    }
                    U[base + k] = s;
                }
            }
            // E | F
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
                T col[nq];
                if (t < cb * nq2) {
                    const auto [e, j, k] = split3<nq, nq>(t);
                    T* er = R + e * ebox + j * pk + k;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        col[n] = er[n * nq * pk];
                    }
                    #pragma unroll
                    for (int i = 0; i < nq; ++i) {
                        T s = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            s += D[n * nq + i] * col[n];
                        }
                        er[i * nq * pk] = s;
                    }
                } else {
                    const int t2 = t - cb * nq2;
                    const auto [e, i, k] = split3<nq, nq>(t2);
                    T* es = S + e * ebox + i * nq * pk + k;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        col[n] = es[n * pk];
                    }
                    #pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        T s = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            s += D[n * nq + j] * col[n];
                        }
                        es[j * pk] = s;
                    }
                }
            }
            #pragma omp for nowait
            for (int x = 0; x < cb * nq3; ++x) {
                const int y = x / nq * pk + x % nq;
                b_out[x] = U[y] + R[y] + S[y];
            }
        }
    }
}

// elements per team of sf: at most `want`, three SLM arrays within 60 KB
constexpr int sf_batch(const int nq, const int want)
{
    const int pk = (nq % 2 == 0) ? nq + 1 : nq;
    const int by_slm = 61440 / (3 * nq * nq * pk * 4);
    const int b = (want < by_slm) ? want : by_slm;
    return (b > 1) ? b : 1;
}
// work-items per team of sf: `width` columns per element (1 or 2) of the
// batch, rounded to whole sub-groups and within 1024
constexpr int sf_block(const int nq, const int batch, const int width)
{
    const int w = round16(width * batch * nq * nq);
    return (w < 1024) ? w : 1024;
}

// ---------------------------------------------------------------------------
// cu: port of the CUDA grid-strided BK5 kernel (LaplaceOperator): a fixed
// grid of teams strides over batches; D in SLM; thread (e, j, k) walks i
// with the r-column in registers, s and t derivatives from SLM per point, G
// factor-major (coalesced over (j, k)); fluxes to SLM; the transposed pass
// per column again.  The CUDA source pairs rqr with qt (Grr * qt + ... +
// Grt * qr); here the fluxes are Grr * qr + Grs * qs + Grt * qt, as in
// BK5.cpp.  Teams: BK_TEAMS (g_grid), 0 = one team per batch.
// ---------------------------------------------------------------------------
inline int g_grid = 0;

template <typename T, int nq, int F, int batch>
void bk5_cu(const std::size_t nelmt, const T* __restrict__ D, const T* __restrict__ G,
            const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int block = round16(batch * nq2);
    static_assert(block <= 1024, "a work-group holds at most 1024 work-items");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;

    #pragma omp target teams num_teams(nteams) thread_limit(block) \
        map(to: D[:nq*nq], G[:nelmt*F*nq3], in[:nelmt*nq3]) map(from: out[:nelmt*nq3])
    {
        T Ds[nq2];
        T wsp[batch * nq3];
        T rqr[batch * nq3];
        T rqs[batch * nq3];
        T rqt[batch * nq3];
        #pragma omp parallel
        {
            #pragma omp for
            for (int t = 0; t < nq2; ++t) {
                Ds[t] = D[t];
            }
            for (int eb = omp_get_team_num(); eb < nbatches; eb += omp_get_num_teams()) {
                const int remaining = ne - eb * batch;
                const int cb = (batch < remaining) ? batch : remaining;
                const T* b_in = in + std::size_t(eb) * (batch * nq3);
                const T* b_G = G + std::size_t(eb) * (batch * F * nq3);
                T* b_out = out + std::size_t(eb) * (batch * nq3);
                #pragma omp for
                for (int x = 0; x < cb * nq3; ++x) {
                    wsp[x] = b_in[x];
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    const auto [e, j, k] = split3<nq, nq>(t);
                    const T* ew = wsp + e * nq3;
                    const T* eG = b_G + e * (F * nq3) + j * nq + k;
                    T r_i[nq];
                    T r_j[nq];
                    T r_k[nq];
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        r_i[n] = ew[n * nq2 + j * nq + k];
                        r_j[n] = Ds[j * nq + n];
                        r_k[n] = Ds[k * nq + n];
                    }
                    #pragma unroll
                    for (int i = 0; i < nq; ++i) {
                        T g[F];
                        #pragma unroll
                        for (int f = 0; f < F; ++f) {
                            g[f] = eG[f * nq3 + i * nq2];
                        }
                        T qr = 0;
                        T qs = 0;
                        T qt = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            qr += Ds[i * nq + n] * r_i[n];
                            qs += r_j[n] * ew[i * nq2 + n * nq + k];
                            qt += r_k[n] * ew[i * nq2 + j * nq + n];
                        }
                        const int x = e * nq3 + i * nq2 + j * nq + k;
                        apply_g<T, F>(g, qr, qs, qt, rqr[x], rqs[x], rqt[x]);
                    }
                }
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    const auto [e, j, k] = split3<nq, nq>(t);
                    T r_i[nq];
                    T r_j[nq];
                    T r_k[nq];
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        r_i[n] = rqr[e * nq3 + n * nq2 + j * nq + k];
                        r_j[n] = Ds[n * nq + j];
                        r_k[n] = Ds[n * nq + k];
                    }
                    #pragma unroll
                    for (int i = 0; i < nq; ++i) {
                        T tmp = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            tmp += r_i[n] * Ds[n * nq + i];
                        }
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            tmp += rqs[e * nq3 + i * nq2 + n * nq + k] * r_j[n];
                        }
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            tmp += rqt[e * nq3 + i * nq2 + j * nq + n] * r_k[n];
                        }
                        b_out[e * nq3 + i * nq2 + j * nq + k] = tmp;
                    }
                }
            }
        }
    }
}

using kernel_fn = void (*)(std::size_t, const float*, const float*, const float*, float*);

struct variant {
    const char* name;
    kernel_fn fn;
    int F;
    int layout;   // 0: factor-major G(f,i,j,k), 1: interleaved G(i,j,f,k), 2: column G(f,k,i,j)
};

#define PT(F, L, IL, B) \
    {"pt" #F #L "_b" #B, &bk5_pt<float, NQ, F, IL, pt_batch(NQ, B)>, F, IL}

// "fill" teams of exactly P work-items (P a power of two): as many whole
// elements as fit, at least one (then the team is the points rounded to 16
// if an element exceeds P)
constexpr int fill_batch(const int nq, const int P) { return (P / (nq * nq * nq) > 1) ? P / (nq * nq * nq) : 1; }
constexpr int fill_group(const int nq, const int P)
{
    return (nq * nq * nq <= P) ? P : round16(nq * nq * nq);
}

#define FILL(F, L, IL, P) \
    {"pt" #F #L "_f" #P, &bk5_pt<float, NQ, F, IL, fill_batch(NQ, P), fill_group(NQ, P)>, F, IL}

#define TID(F, L, IL, B, EARLY, X) \
    {"tid" #F #L #X "_b" #B, &bk5_tid<float, NQ, F, IL, pt_batch(NQ, B), EARLY>, F, IL}

#define SF(F, B, W) \
    {"sf" #F "c_b" #B "x" #W, &bk5_sf<float, NQ, F, sf_batch(NQ, B), sf_block(NQ, sf_batch(NQ, B), W)>, F, 2}

#define CU(F, B) \
    {"cu" #F "f_b" #B, &bk5_cu<float, NQ, F, pt_batch(NQ, B)>, F, 0}

// in the order they run
const variant variants[] = {
#ifndef BK_NO_BASELINES
    // one element per thread: baselines, ~30x slower than the best variants
    // on the A770 at p = 6 (the check is the host reference)
    {"ref6i", &bk5_ref<float, NQ, 6, true>, 6, true},
    {"ref9i", &bk5_ref<float, NQ, 9, true>, 9, true},
#endif
    PT(6, i, true, 1), PT(6, i, true, 2), PT(6, i, true, 4), PT(6, i, true, 8), PT(6, i, true, 16),
    PT(9, i, true, 1), PT(9, i, true, 2), PT(9, i, true, 4), PT(9, i, true, 8), PT(9, i, true, 16),
    PT(6, f, false, 1), PT(6, f, false, 2), PT(6, f, false, 4), PT(6, f, false, 8), PT(6, f, false, 16),
    PT(9, f, false, 1), PT(9, f, false, 2), PT(9, f, false, 4), PT(9, f, false, 8), PT(9, f, false, 16),
    FILL(6, i, true, 512), FILL(6, f, false, 512), FILL(9, i, true, 512), FILL(9, f, false, 512),
    FILL(6, i, true, 1024), FILL(6, f, false, 1024), FILL(9, i, true, 1024), FILL(9, f, false, 1024),
    TID(6, i, true, 1, true, e), TID(6, f, false, 1, true, e), TID(9, i, true, 1, true, e), TID(9, f, false, 1, true, e),
    TID(6, i, true, 2, true, e), TID(6, f, false, 2, true, e), TID(9, i, true, 2, true, e), TID(9, f, false, 2, true, e),
    TID(6, i, true, 4, true, e), TID(6, f, false, 4, true, e), TID(9, i, true, 4, true, e), TID(9, f, false, 4, true, e),
    TID(6, i, true, 1, false, l), TID(6, f, false, 1, false, l), TID(9, i, true, 1, false, l), TID(9, f, false, 1, false, l),
    SF(6, 1, 1), SF(6, 2, 1), SF(6, 4, 1), SF(6, 8, 1), SF(6, 16, 1), SF(6, 32, 1),
    SF(6, 1, 2), SF(6, 2, 2), SF(6, 4, 2), SF(6, 8, 2), SF(6, 16, 2), SF(6, 32, 2),
    SF(9, 1, 1), SF(9, 2, 1), SF(9, 4, 1), SF(9, 8, 1), SF(9, 16, 1), SF(9, 32, 1),
    SF(9, 1, 2), SF(9, 2, 2), SF(9, 4, 2), SF(9, 8, 2), SF(9, 16, 2), SF(9, 32, 2),
    CU(6, 1), CU(6, 2), CU(6, 4), CU(6, 8), CU(6, 16), CU(9, 1), CU(9, 2), CU(9, 4), CU(9, 8), CU(9, 16),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nq3 = nq * nq * nq;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    g_grid = bk::env_int("BK_TEAMS", 0);

    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nq3);
    std::vector<float> G6i(nelmt * 6 * nq3);
    bk::fill_random_test_data(in, G6i);   // in in [-1, 1], G6i = 1.5 + [-1, 1]
    // BK_CHECK=0: benchmark only, no reference
    const bk::reference_check check(in.size(), [&](std::vector<double>& ref) {
        bk::reference_bk5(nq, nelmt, D, G6i, in, ref);
    });
    std::vector<float> out(in.size());
    std::printf("BK5 offload: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d\n", nq - 1, nq, nelmt,
                nelmt * nq3, ntests);

    // G of the variant: G6i itself (6 factors interleaved), or built from it
    // and kept while the next variants use the same (F, layout)
    std::vector<float> G_built;
    std::pair<int, int> built_key{0, -1};
    const float* d_D = D.data();
    const float* d_in = in.data();
    float* d_out = out.data();
    const std::size_t n_in = in.size();
    #pragma omp target data map(to: d_D[:nq*nq], d_in[:n_in]) map(alloc: d_out[:n_in])
    {
        for (const variant& v : variants) {
            if (!bk::selected(v.name)) {
                continue;
            }
            const std::pair<int, int> key{v.F, v.layout};
            if (key != std::make_pair(6, 1) && key != built_key) {
                G_built = bk::make_bk5_G(G6i, nelmt, nq, v.F, v.layout);
                built_key = key;
            }
            const std::vector<float>& G = (key == std::make_pair(6, 1)) ? G6i : G_built;
            const float* d_G = G.data();
            const std::size_t n_G = G.size();
            double best = 0.0;
            #pragma omp target data map(to: d_G[:n_G])
            {
                v.fn(nelmt, d_D, d_G, d_in, d_out);   // warm-up
                // timed: ntests launches of the kernel, the best wall time
                best = bk::time_min(ntests, [&] {
                    v.fn(nelmt, d_D, d_G, d_in, d_out);
                });
            }
            const std::string chk = check.text([&]() -> const std::vector<float>& {
                #pragma omp target update from(d_out[:n_in])
                return out;
            });
            const double bytes = sizeof(float) * double(nelmt) * nq3 * (2.0 + v.F);
            std::printf("%-10s GDoF/s = %7.3f  GB/s = %6.1f  time = %8.3f ms  relerr = %s\n", v.name,
                        1e-9 * n_in / best, 1e-9 * bytes / best, 1e3 * best, chk.c_str());
        }
    }
    return 0;
}
