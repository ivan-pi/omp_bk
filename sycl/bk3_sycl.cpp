// bk3_sycl.cpp -- BK3 (Poisson operator) in SYCL: the port of the CUDA
// kernel (cu) and sum-factorised sweeps over register columns (sf), a batch
// of elements per work-group, G in BK3's factor-major layout G(e, f, p, q, r).
//
//   scripts/build.sh bk3_sycl_<p>       (-> build/a770/bk3_sycl_<p>, nq = p + 2)
//   scripts/run.sh bk3_sycl_<p> [nelmt] [ntests]
//   BK_KERNEL="sf_b4x1 cu_b1" ...      run a subset (default: all)
//   BK_TEAMS=n                          cu: n work-groups stride over the batches
//                                       (default: one per batch)
//
// The operator (openmp/bk3_sumfact.h): interpolate in(e,i,j,k) to the
// quadrature points with B(i,p), derivatives with D(n,p), fluxes with the
// six symmetric factors (rr = Grr qr + Grs qs + Grt qt, ...), transposed
// derivatives with D(p,n), interpolate back.  The CUDA source pairs rr with
// qt (Grr * qt + ... + Grt * qr); here the fluxes are those of BK3.cpp.
//
// Variants:
//   cu_b<batch>   CUDA port: basis and D copied to local memory, 12 steps
//                 through four dense nq^3 boxes per element, one work-item
//                 per (row, column) of a step, a barrier after each step.
//   sf_b<batch>x<w>  three local boxes U, R, S per element, nq x nq rows of
//                 pc floats (pc = nq, or nq + 1 for even nq, so a column
//                 sweep along the last index hits distinct banks); w
//                 work-items per (q, r) column of the batch:
//       copy  in -> S(i,j,k)
//       1     thread (i,j): R(i,j,r) = B along k
//       2     thread (i,r): S(i,q,r) = B along j
//       3     thread (q,r): U(p,q,r) = B along i
//       A | B thread (p,r): R = D along q;  thread (p,q): S = D along r
//       C     thread (q,r), column along p: qr = D along p, G(f,p,q,r)
//             (coalesced over (q,r)), fluxes: rs -> R, rt -> S in place, rr
//             in registers; U(:,q,r) = D^T rr
//       E | F thread (p,r): R = D^T along q;  thread (p,q): S = D^T along r
//       7     thread (q,r): U(i,q,r) = B^T (U + R + S) along p
//       8     thread (i,r): R(i,j,r) = B^T along q
//       9     thread (i,j): S(i,j,k) = B^T along r
//       copy  S -> out
//   sfr_b<batch>  sf with one work-item per (q, r) column that carries the
//                 column of u (3 -> C) and D^T rr (C -> 7) in registers
//   sfp_b<batch>x1, sfrp_b<batch>  sf and sfr with the box strides of the
//                 fewest SLM bank conflicts (box_padded)
//   sfr_f<P>_sg8  sfr in fill groups: work-groups of exactly P work-items
//                 holding as many whole elements as fit
//   suffix _sg8   sub-group size 8 required (SIMD8: twice the registers per
//                 work-item; IGC picks SIMD16 for these kernels otherwise, and
//                 SIMD8 for the OpenMP port of cu); sf variants then also hold
//                 B and D in registers
//
// Data: pseudo-random in and G (bk::fill_random_test_data), B(i,p) =
// cos(i * nq + p), D(n,p) = cos(n * nq + p).  Check: largest difference to
// the double-precision CPU reference relative to its largest |value|.
// GB/s counts in + out + G (6 floats per point).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"

#ifndef NQ
#define NQ 6
#endif

namespace {

constexpr int round16(const int n) { return bk::round_up(n, 16); }

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

// fluxes from the derivatives at a point, the six symmetric factors
template <typename T>
inline void apply_g(const T (&g)[6], const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    rr = g[0] * qr + g[1] * qs + g[2] * qt;
    rs = g[1] * qr + g[3] * qs + g[4] * qt;
    rt = g[2] * qr + g[4] * qs + g[5] * qt;
}

// ---------------------------------------------------------------------------
// cu: the CUDA kernel
// ---------------------------------------------------------------------------
constexpr int cu_batch(const int nq, const int want)
{
    const int by_slm = (61440 / 4 - nq * nq * 2) / (4 * nq * nq * nq);
    const int by_threads = 1024 / (nq * nq);
    int b = want;
    if (b > by_slm) {
        b = by_slm;
    }
    if (b > by_threads) {
        b = by_threads;
    }
    return (b > 1) ? b : 1;
}

template <typename T, int nq, int batch>
struct cu_kernel {
    static constexpr int nm = nq - 1;
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;
    static constexpr int nm3 = nm * nm * nm;
    static constexpr int lmem = nm * nq + nq2 + 4 * batch * nq3;
    static_assert(lmem * int(sizeof(T)) <= 61440, "local memory within 60 KB");
    using local = sycl::local_accessor<T, 1>;

    const T* basis;
    const T* dbasis;
    const T* G;
    const T* in;
    T* out;
    int nelmt;
    // one accessor per array: the compiler then knows that a store to a
    // work array leaves the others (the basis above all) unchanged
    local l_basis;
    local l_dbasis;
    local l_wsp0;
    local l_wsp1;
    local l_rqr;
    local l_rqs;

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const auto ptr = [](const local& a) { return a.template get_multi_ptr<sycl::access::decorated::no>().get(); };
        T* s_basis = ptr(l_basis);
        T* s_dbasis = ptr(l_dbasis);
        T* s_wsp0 = ptr(l_wsp0);
        T* s_wsp1 = ptr(l_wsp1);
        T* s_rqr = ptr(l_rqr);
        T* s_rqs = ptr(l_rqs);
        T* s_rqt = s_wsp0;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };
        for (int t = lid; t < nm * nq; t += lsz) {
            s_basis[t] = basis[t];
        }
        for (int t = lid; t < nq2; t += lsz) {
            s_dbasis[t] = dbasis[t];
        }
        bar();
        const int nbatches = (nelmt + batch - 1) / batch;
        for (int eb = static_cast<int>(it.get_group(0)); eb < nbatches; eb += static_cast<int>(it.get_group_range(0))) {
            const int remaining = nelmt - eb * batch;
            const int cb = (batch < remaining) ? batch : remaining;
            const T* b_in = in + std::size_t(eb) * (batch * nm3);
            const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
            T* b_out = out + std::size_t(eb) * (batch * nm3);
            // 1: in -> wsp0
            for (int x = lid; x < cb * nm3; x += lsz) {
                s_wsp0[x] = b_in[x];
            }
            bar();
            // 2: direction 0, wsp0(e,i,j,k) -> wsp1(e,p,j,k)
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                const auto [e, j, k] = split3<nm, nm>(t);
                T r[nm];
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    r[i] = s_wsp0[e * nm3 + (i * nm + j) * nm + k];
                }
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    T s = 0;
                    #pragma unroll
                    for (int i = 0; i < nm; ++i) {
                        s += s_basis[i * nq + p] * r[i];
                    }
                    s_wsp1[e * nq * nm * nm + (p * nm + j) * nm + k] = s;
                }
            }
            bar();
            // 3: direction 1, wsp1(e,p,j,k) -> wsp0(e,q,p,k)
            for (int t = lid; t < cb * nq * nm; t += lsz) {
                const auto [e, p, k] = split3<nq, nm>(t);
                T r[nm];
                #pragma unroll
                for (int j = 0; j < nm; ++j) {
                    r[j] = s_wsp1[e * nq * nm * nm + (p * nm + j) * nm + k];
                }
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    T s = 0;
                    #pragma unroll
                    for (int j = 0; j < nm; ++j) {
                        s += s_basis[j * nq + q] * r[j];
                    }
                    s_wsp0[e * nq2 * nm + (q * nq + p) * nm + k] = s;
                }
            }
            bar();
            // 4: direction 2, wsp0(e,q,p,k) -> wsp1(e,r,q,p) = u(p,q,r)
            for (int t = lid; t < cb * nq2; t += lsz) {
                const auto [e, q, p] = split3<nq, nq>(t);
                T r[nm];
                #pragma unroll
                for (int k = 0; k < nm; ++k) {
                    r[k] = s_wsp0[e * nq2 * nm + (q * nq + p) * nm + k];
                }
                #pragma unroll
                for (int rr = 0; rr < nq; ++rr) {
                    T s = 0;
                    #pragma unroll
                    for (int k = 0; k < nm; ++k) {
                        s += s_basis[k * nq + rr] * r[k];
                    }
                    s_wsp1[e * nq3 + (rr * nq + q) * nq + p] = s;
                }
            }
            bar();
            // 5-7: thread (e,q,r) walks p: derivatives, G (coalesced over r), fluxes
            for (int t = lid; t < cb * nq2; t += lsz) {
                const auto [e, q, r] = split3<nq, nq>(t);
                const T* u = s_wsp1 + e * nq3;   // u(p,q,r) at (r * nq + q) * nq + p
                const T* eG = b_G + e * 6 * nq3 + q * nq + r;
                T r_p[nq];
                T r_q[nq];
                T r_r[nq];
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    r_p[n] = u[(r * nq + q) * nq + n];
                    r_q[n] = s_dbasis[n * nq + q];
                    r_r[n] = s_dbasis[n * nq + r];
                }
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    T qr = 0;
                    T qs = 0;
                    T qt = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        qr += s_dbasis[n * nq + p] * r_p[n];
                        qs += r_q[n] * u[(r * nq + n) * nq + p];
                        qt += r_r[n] * u[(n * nq + q) * nq + p];
                    }
                    T g[6];
                    #pragma unroll
                    for (int f = 0; f < 6; ++f) {
                        g[f] = eG[f * nq3 + p * nq2];
                    }
                    const int x = e * nq3 + (p * nq + q) * nq + r;
                    apply_g(g, qr, qs, qt, s_rqr[x], s_rqs[x], s_rqt[x]);
                }
            }
            bar();
            // 8: transposed derivatives, thread (e,q,r) walks p -> wsp1(e,r,q,p)
            for (int t = lid; t < cb * nq2; t += lsz) {
                const auto [e, q, r] = split3<nq, nq>(t);
                const int eo = e * nq3;
                T r_p[nq];
                T r_q[nq];
                T r_r[nq];
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    r_p[n] = s_rqr[eo + (n * nq + q) * nq + r];
                    r_q[n] = s_dbasis[q * nq + n];
                    r_r[n] = s_dbasis[r * nq + n];
                }
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += r_p[n] * s_dbasis[p * nq + n];
                    }
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += s_rqs[eo + (p * nq + n) * nq + r] * r_q[n];
                    }
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += s_rqt[eo + (p * nq + q) * nq + n] * r_r[n];
                    }
                    s_wsp1[eo + (r * nq + q) * nq + p] = s;
                }
            }
            bar();
            // 9: direction 2, wsp1(e,r,q,p) -> wsp0(e,k,q,p)
            for (int t = lid; t < cb * nq2; t += lsz) {
                const auto [e, q, p] = split3<nq, nq>(t);
                T r[nq];
                #pragma unroll
                for (int rr = 0; rr < nq; ++rr) {
                    r[rr] = s_wsp1[e * nq3 + (rr * nq + q) * nq + p];
                }
                #pragma unroll
                for (int k = 0; k < nm; ++k) {
                    T s = 0;
                    #pragma unroll
                    for (int rr = 0; rr < nq; ++rr) {
                        s += s_basis[k * nq + rr] * r[rr];
                    }
                    s_wsp0[e * nm * nq2 + (k * nq + q) * nq + p] = s;
                }
            }
            bar();
            // 10: direction 1, wsp0(e,k,q,p) -> wsp1(e,k,j,p)
            for (int t = lid; t < cb * nm * nq; t += lsz) {
                const auto [e, k, p] = split3<nm, nq>(t);
                T r[nq];
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    r[q] = s_wsp0[e * nm * nq2 + (k * nq + q) * nq + p];
                }
                #pragma unroll
                for (int j = 0; j < nm; ++j) {
                    T s = 0;
                    #pragma unroll
                    for (int q = 0; q < nq; ++q) {
                        s += s_basis[j * nq + q] * r[q];
                    }
                    s_wsp1[e * nm * nm * nq + (k * nm + j) * nq + p] = s;
                }
            }
            bar();
            // 11: direction 0, wsp1(e,k,j,p) -> wsp0(e,i,j,k)
            for (int t = lid; t < cb * nm * nm; t += lsz) {
                const auto [e, j, k] = split3<nm, nm>(t);
                T r[nq];
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    r[p] = s_wsp1[e * nm * nm * nq + (k * nm + j) * nq + p];
                }
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    T s = 0;
                    #pragma unroll
                    for (int p = 0; p < nq; ++p) {
                        s += s_basis[i * nq + p] * r[p];
                    }
                    s_wsp0[e * nm3 + (i * nm + j) * nm + k] = s;
                }
            }
            bar();
            // 12: wsp0 -> out
            for (int x = lid; x < cb * nm3; x += lsz) {
                b_out[x] = s_wsp0[x];
            }
            bar();
        }
    }
};

// ---------------------------------------------------------------------------
// sf: register-column sweeps through three padded boxes
// ---------------------------------------------------------------------------
// Box layout: (e, a, b, c) at e * se + a * sa + b * sb + c.
struct box_strides {
    int sb;
    int sa;
    int se;
};

// dense: the rows padded to an odd length (sb = nq, or nq + 1 for even nq)
constexpr box_strides box_dense(const int nq)
{
    const int pc = (nq % 2 == 0) ? nq + 1 : nq;
    return {pc, nq * pc, nq * nq * pc};
}

// padded: the strides with the fewest SLM bank conflicts over all steps for
// SIMD8 messages (scripts/study/bk3_slm_pad.py)
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

// elements per work-group: at most `want`, three boxes within 60 KB and
// `width` work-items per (q, r) column within 1024
constexpr int sf_batch(const int nq, const int want, const int width, const bool padded = false)
{
    const int by_slm = 61440 / (3 * box_layout(nq, padded).se * 4);
    const int by_threads = 1024 / (width * nq * nq);
    int b = want;
    if (b > by_slm) {
        b = by_slm;
    }
    if (b > by_threads) {
        b = by_threads;
    }
    return (b > 1) ? b : 1;
}

// fill groups (the oneAPI GPU guide's occupancy rule): a work-group of
// exactly P work-items (a power of two, so its hardware threads divide an
// Xe-core's) holding as many whole elements as fit, one work-item per column
constexpr int fill_batch(const int nq, const int P) { return sf_batch(nq, P / (nq * nq), 1); }
constexpr int fill_block(const int nq, const int P) { return (nq * nq <= P) ? P : round16(nq * nq); }

constexpr int sf_block(const int nq, const int batch, const int width)
{
    const int w = round16(width * batch * nq * nq);
    return (w < 1024) ? w : 1024;
}

// carry: one work-item per (q, r) column (block >= batch * nq^2), which
// keeps its column of u from step 3 to C and its D^T rr from C to step 7 in
// registers instead of U
// padded: box_padded strides instead of box_dense.  breg: B and D loaded
// once into registers (uniform values, scalar operands of the FMAs) instead
// of read from global memory in every step; pays off at SIMD8, where a
// register holds 8 floats, and spills at SIMD16
template <typename T, int nq, int batch, bool carry = false, bool padded = false, bool breg = false>
struct sf_kernel {
    static constexpr int nm = nq - 1;
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;
    static constexpr int nm3 = nm * nm * nm;
    static constexpr box_strides L = box_layout(nq, padded);
    static constexpr int ebox = L.se;
    static constexpr int lmem = 3 * batch * ebox;
    static_assert(lmem * int(sizeof(T)) <= 61440, "three local boxes within 60 KB");
    using local = sycl::local_accessor<T, 1>;

    const T* gB;
    const T* gD;
    const T* G;
    const T* in;
    T* out;
    int nelmt;
    // one accessor per box, so the compiler knows the boxes do not alias
    local lU;
    local lR;
    local lS;

    // box index of (a, b, c) of element e
    static constexpr int at(const int e, const int a, const int b, const int c) { return e * L.se + a * L.sa + b * L.sb + c; }

    [[intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const
    {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lsz = static_cast<int>(it.get_local_range(0));
        const int eb = static_cast<int>(it.get_group(0));
        const int remaining = nelmt - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);
        const auto ptr = [](const local& a) { return a.template get_multi_ptr<sycl::access::decorated::no>().get(); };
        T* U = ptr(lU);
        T* R = ptr(lR);
        T* S = ptr(lS);
        T Bv[breg ? nm * nq : 1];
        T Dv[breg ? nq2 : 1];
        if constexpr (breg) {
            #pragma unroll
            for (int x = 0; x < nm * nq; ++x) {
                Bv[x] = gB[x];
            }
            #pragma unroll
            for (int x = 0; x < nq2; ++x) {
                Dv[x] = gD[x];
            }
        }
        const T* B = breg ? Bv : gB;
        const T* D = breg ? Dv : gD;
        const auto bar = [&]() { it.barrier(sycl::access::fence_space::local_space); };

        for (int x = lid; x < cb * nm3; x += lsz) {
            const int e = x / nm3;
            const int y = x % nm3;
            S[at(e, y / (nm * nm), y / nm % nm, y % nm)] = b_in[x];
        }
        bar();
        // 1: along k
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const auto [e, i, j] = split3<nm, nm>(t);
            T col[nm];
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                col[k] = S[at(e, i, j, k)];
            }
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                T s = 0;
                #pragma unroll
                for (int k = 0; k < nm; ++k) {
                    s += B[k * nq + r] * col[k];
                }
                R[at(e, i, j, r)] = s;
            }
        }
        bar();
        // 2: along j
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            const auto [e, i, r] = split3<nm, nq>(t);
            T col[nm];
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                col[j] = R[at(e, i, j, r)];
            }
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                T s = 0;
                #pragma unroll
                for (int j = 0; j < nm; ++j) {
                    s += B[j * nq + q] * col[j];
                }
                S[at(e, i, q, r)] = s;
            }
        }
        bar();
        // the (q, r) column a work-item carries in registers (carry only)
        T ucol[nq];
        // 3: along i
        for (int t = lid; t < cb * nq2; t += lsz) {
            const auto [e, q, r] = split3<nq, nq>(t);
            T col[nm];
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                col[i] = S[at(e, i, q, r)];
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                T s = 0;
                #pragma unroll
                for (int i = 0; i < nm; ++i) {
                    s += B[i * nq + p] * col[i];
                }
                U[at(e, p, q, r)] = s;
                ucol[p] = s;
            }
        }
        bar();
        // A | B: derivatives along q and r
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            T col[nq];
            if (t < cb * nq2) {
                const auto [e, p, r] = split3<nq, nq>(t);
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = U[at(e, p, n, r)];
                }
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[n * nq + q] * col[n];
                    }
                    R[at(e, p, q, r)] = s;
                }
            } else {
                const auto [e, p, q] = split3<nq, nq>(t - cb * nq2);
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = U[at(e, p, q, n)];
                }
                #pragma unroll
                for (int r = 0; r < nq; ++r) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[n * nq + r] * col[n];
                    }
                    S[at(e, p, q, r)] = s;
                }
            }
        }
        bar();
        // C: column along p, G, fluxes
        for (int t = lid; t < cb * nq2; t += lsz) {
            const auto [e, q, r] = split3<nq, nq>(t);
            const T* eG = b_G + e * 6 * nq3 + q * nq + r;
            T col[nq];
            T fr[nq];
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                col[n] = carry ? ucol[n] : U[at(e, n, q, r)];
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                T qr = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    qr += D[n * nq + p] * col[n];
                }
                T g[6];
                #pragma unroll
                for (int f = 0; f < 6; ++f) {
                    g[f] = eG[f * nq3 + p * nq2];
                }
                const int x = at(e, p, q, r);
                T rs;
                T rt;
                apply_g(g, qr, R[x], S[x], fr[p], rs, rt);
                R[x] = rs;
                S[x] = rt;
            }
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                T s = 0;
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    s += D[p * nq + n] * fr[n];
                }
                if constexpr (carry) {
                    ucol[p] = s;
                } else {
                    U[at(e, p, q, r)] = s;
                }
            }
        }
        bar();
        // E | F: transposed derivatives along q and r, in place
        for (int t = lid; t < 2 * cb * nq2; t += lsz) {
            T col[nq];
            if (t < cb * nq2) {
                const auto [e, p, r] = split3<nq, nq>(t);
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = R[at(e, p, n, r)];
                }
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[q * nq + n] * col[n];
                    }
                    R[at(e, p, q, r)] = s;
                }
            } else {
                const auto [e, p, q] = split3<nq, nq>(t - cb * nq2);
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = S[at(e, p, q, n)];
                }
                #pragma unroll
                for (int r = 0; r < nq; ++r) {
                    T s = 0;
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        s += D[r * nq + n] * col[n];
                    }
                    S[at(e, p, q, r)] = s;
                }
            }
        }
        bar();
        // 7: sum and back along p, in place in U's column
        for (int t = lid; t < cb * nq2; t += lsz) {
            const auto [e, q, r] = split3<nq, nq>(t);
            T w[nq];
            #pragma unroll
            for (int p = 0; p < nq; ++p) {
                const int x = at(e, p, q, r);
                w[p] = (carry ? ucol[p] : U[x]) + R[x] + S[x];
            }
            #pragma unroll
            for (int i = 0; i < nm; ++i) {
                T s = 0;
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    s += B[i * nq + p] * w[p];
                }
                U[at(e, i, q, r)] = s;
            }
        }
        bar();
        // 8: back along q
        for (int t = lid; t < cb * nm * nq; t += lsz) {
            const auto [e, i, r] = split3<nm, nq>(t);
            T col[nq];
            #pragma unroll
            for (int q = 0; q < nq; ++q) {
                col[q] = U[at(e, i, q, r)];
            }
            #pragma unroll
            for (int j = 0; j < nm; ++j) {
                T s = 0;
                #pragma unroll
                for (int q = 0; q < nq; ++q) {
                    s += B[j * nq + q] * col[q];
                }
                R[at(e, i, j, r)] = s;
            }
        }
        bar();
        // 9: back along r
        for (int t = lid; t < cb * nm * nm; t += lsz) {
            const auto [e, i, j] = split3<nm, nm>(t);
            T col[nq];
            #pragma unroll
            for (int r = 0; r < nq; ++r) {
                col[r] = R[at(e, i, j, r)];
            }
            #pragma unroll
            for (int k = 0; k < nm; ++k) {
                T s = 0;
                #pragma unroll
                for (int r = 0; r < nq; ++r) {
                    s += B[k * nq + r] * col[r];
                }
                S[at(e, i, j, k)] = s;
            }
        }
        bar();
        for (int x = lid; x < cb * nm3; x += lsz) {
            const int e = x / nm3;
            const int y = x % nm3;
            b_out[x] = S[at(e, y / (nm * nm), y / nm % nm, y % nm)];
        }
    }
};

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
struct device_data {
    float* basis;
    float* D;
    float* G;
    float* in;
    float* out;
    int nelmt;
};

using launch_fn = sycl::event (*)(sycl::queue&, const device_data&);

// a kernel with the work-group size wg fixed at compile time (the oneAPI
// GPU guide's reqd_work_group_size: get_local_range() becomes a constant)
// and a required sub-group size (SIMD width) of sg, 0: the compiler's choice
template <typename K, int sg, int wg>
struct with_sg {
    K k;
    [[sycl::reqd_work_group_size(wg), intel::kernel_args_restrict]] void operator()(sycl::nd_item<1> it) const { k(it); }
};
template <typename K, int wg>
struct with_sg<K, 8, wg> {
    K k;
    [[sycl::reqd_work_group_size(wg), sycl::reqd_sub_group_size(8), intel::kernel_args_restrict]] void operator()(
        sycl::nd_item<1> it) const
    {
        k(it);
    }
};

inline int g_grid = 0;   // BK_TEAMS

template <int batch, int sg>
sycl::event launch_cu(sycl::queue& q, const device_data& d)
{
    using K = cu_kernel<float, NQ, batch>;
    constexpr int block = round16(batch * NQ * NQ);
    static_assert(block <= 1024, "at most 1024 work-items");
    const int nbatches = (d.nelmt + batch - 1) / batch;
    const int ngroups = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;
    const sycl::nd_range<1> range{std::size_t(ngroups) * block, std::size_t(block)};
    return q.submit([&](sycl::handler& cgh) {
        using L = sycl::local_accessor<float, 1>;
        constexpr int nq3 = NQ * NQ * NQ;
        const K k{d.basis, d.D, d.G, d.in, d.out, d.nelmt,
                  L(sycl::range<1>((NQ - 1) * NQ), cgh), L(sycl::range<1>(NQ * NQ), cgh),
                  L(sycl::range<1>(batch * nq3), cgh), L(sycl::range<1>(batch * nq3), cgh),
                  L(sycl::range<1>(batch * nq3), cgh), L(sycl::range<1>(batch * nq3), cgh)};
        cgh.parallel_for(range, with_sg<K, sg, block>{k});
    });
}

template <int batch, int block, int sg, bool carry = false, bool padded = false>
sycl::event launch_sf(sycl::queue& q, const device_data& d)
{
    using K = sf_kernel<float, NQ, batch, carry, padded, sg == 8>;
    static_assert(!carry || block >= batch * NQ * NQ, "carry: one work-item per column");
    static_assert(block <= 1024 && block % 16 == 0, "whole sub-groups, at most 1024 work-items");
    const int ngroups = (d.nelmt + batch - 1) / batch;
    const sycl::nd_range<1> range{std::size_t(ngroups) * block, std::size_t(block)};
    return q.submit([&](sycl::handler& cgh) {
        using L = sycl::local_accessor<float, 1>;
        const K k{d.basis, d.D, d.G, d.in, d.out, d.nelmt, L(sycl::range<1>(batch * K::ebox), cgh),
                  L(sycl::range<1>(batch * K::ebox), cgh), L(sycl::range<1>(batch * K::ebox), cgh)};
        cgh.parallel_for(range, with_sg<K, sg, block>{k});
    });
}

struct variant {
    const char* name;
    launch_fn fn;
};

// suffix _sg<n>: required sub-group size n
#define CU(B, SG, SUFFIX) {"cu_b" #B SUFFIX, &launch_cu<cu_batch(NQ, B), SG>}
#define SF(B, W, SG, SUFFIX) {"sf_b" #B "x" #W SUFFIX, &launch_sf<sf_batch(NQ, B, W), sf_block(NQ, sf_batch(NQ, B, W), W), SG>}
#define SFP(B, SG, SUFFIX) \
    {"sfp_b" #B "x1" SUFFIX, &launch_sf<sf_batch(NQ, B, 1, true), sf_block(NQ, sf_batch(NQ, B, 1, true), 1), SG, false, true>}
#define SFRP(B, SG, SUFFIX) \
    {"sfrp_b" #B SUFFIX, &launch_sf<sf_batch(NQ, B, 1, true), sf_block(NQ, sf_batch(NQ, B, 1, true), 1), SG, true, true>}
#define SFF(P) {"sfr_f" #P "_sg8", &launch_sf<fill_batch(NQ, P), fill_block(NQ, P), 8, true>}
#define SFR(B, SG, SUFFIX) {"sfr_b" #B SUFFIX, &launch_sf<sf_batch(NQ, B, 1), sf_block(NQ, sf_batch(NQ, B, 1), 1), SG, true>}

const variant variants[] = {
    CU(1, 0, ""), CU(2, 0, ""), CU(4, 0, ""), CU(8, 0, ""), CU(16, 0, ""),
    CU(1, 8, "_sg8"), CU(2, 8, "_sg8"), CU(4, 8, "_sg8"), CU(8, 8, "_sg8"), CU(16, 8, "_sg8"),
    SF(1, 1, 0, ""), SF(2, 1, 0, ""), SF(4, 1, 0, ""), SF(8, 1, 0, ""), SF(16, 1, 0, ""), SF(32, 1, 0, ""),
    SF(1, 2, 0, ""), SF(2, 2, 0, ""), SF(4, 2, 0, ""), SF(8, 2, 0, ""), SF(16, 2, 0, ""),
    SF(1, 1, 8, "_sg8"), SF(2, 1, 8, "_sg8"), SF(4, 1, 8, "_sg8"), SF(8, 1, 8, "_sg8"), SF(16, 1, 8, "_sg8"),
    SF(1, 2, 8, "_sg8"), SF(2, 2, 8, "_sg8"),
    SFR(1, 0, ""), SFR(2, 0, ""), SFR(4, 0, ""), SFR(1, 8, "_sg8"), SFR(2, 8, "_sg8"), SFR(4, 8, "_sg8"),
    SFP(1, 8, "_sg8"), SFP(2, 8, "_sg8"), SFP(4, 8, "_sg8"), SFRP(1, 8, "_sg8"), SFRP(2, 8, "_sg8"),
    SFF(128), SFF(256), SFF(512),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    g_grid = bk::env_int("BK_TEAMS", 0);

    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}}};
    std::printf("BK3 SYCL: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d, device %s\n", nq - 2, nq, nelmt,
                nelmt * nm * nm * nm, ntests, q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    std::vector<float> G(nelmt * 6 * nq * nq * nq);
    bk::fill_random_test_data(in, G);   // in in [-1, 1], G = 1.5 + [-1, 1]
    std::vector<double> ref(in.size());
    bk::reference_bk3(nq, nelmt, basis, D, G, in, ref);
    const double big = bk::max_abs(ref);

    device_data d{sycl::malloc_device<float>(basis.size(), q), sycl::malloc_device<float>(D.size(), q),
                  sycl::malloc_device<float>(G.size(), q), sycl::malloc_device<float>(in.size(), q),
                  sycl::malloc_device<float>(in.size(), q), static_cast<int>(nelmt)};
    q.copy(basis.data(), d.basis, basis.size());
    q.copy(D.data(), d.D, D.size());
    q.copy(G.data(), d.G, G.size());
    q.copy(in.data(), d.in, in.size());
    q.wait_and_throw();

    const double bytes = sizeof(float) * (2.0 * in.size() + G.size());
    std::vector<float> out(in.size());
    for (const variant& v : variants) {
        if (!bk::selected(v.name)) {
            continue;
        }
        q.memset(d.out, 0, out.size() * sizeof(float)).wait();
        const bk::launch_times t = bk::time_launches(ntests, [&] { return v.fn(q, d); });
        q.copy(d.out, out.data(), out.size()).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        std::printf("%-10s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s\n",
                    v.name, 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, relerr, bk::fail_mark(relerr));
    }
    sycl::free(d.basis, q);
    sycl::free(d.D, q);
    sycl::free(d.G, q);
    sycl::free(d.in, q);
    sycl::free(d.out, q);
    return 0;
}
