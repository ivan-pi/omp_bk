// bk3_offload.cpp -- BK3 (Poisson operator) kernel variants in OpenMP
// offload, one order per build (-DNQ=n, nq = p + 2), every variant timed
// and checked against the double-precision CPU reference.  The kernels are
// those of sycl/bk3_sycl.cpp (see its header for the steps), with the work
// arrays at team scope and an `omp for` per step.  icpx places team-scope
// arrays in SLM (zeinfo slm_size); the explicit `omp allocate(...)
// allocator(omp_pteam_mem_alloc)` is rejected for spir64_gen by icpx 2026.1
// ("allocate directive not supported for selected target").
//
//   scripts/build.sh bk3_offload_<p>       (-> build/a770/bk3_offload_<p>)
//   scripts/run.sh bk3_offload_<p> [nelmt] [ntests]
//   BK_KERNEL="sf_b1x1 cu_b2" ...          run a subset (default: all)
//   BK_TEAMS=n                             cu: n teams stride over the batches
//                                          (default: one team per batch)
//
// Variants:
//   ref           BK3.cpp's kernel (openmp/bk3_sumfact.h): one element per
//                 iteration of a teams loop, private work arrays
//   cu_b<batch>   port of the CUDA kernel, a batch of elements per team
//   sf_b<batch>x<w>  register-column sweeps through three padded SLM boxes
//   suffix r      the basis and D held in registers by every thread (uniform
//                 values) instead of read from SLM (cu) or global memory (sf)
//
// Data: pseudo-random in and G (bk::fill_random_test_data), G factor-major
// G(e, f, p, q, r).  GB/s counts in + out + G (6 floats per point).

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk3_sumfact.h"

#ifndef NQ
#define NQ 6
#endif

namespace {

constexpr int round16(const int n) { return bk::round_up(n, 16); }

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

template <typename T>
inline void apply_g(const T (&g)[6], const T qr, const T qs, const T qt, T& rr, T& rs, T& rt)
{
    rr = g[0] * qr + g[1] * qs + g[2] * qt;
    rs = g[1] * qr + g[3] * qs + g[4] * qt;
    rt = g[2] * qr + g[4] * qs + g[5] * qt;
}

// ---------------------------------------------------------------------------
// cu: the CUDA kernel, a fixed grid of teams strides over the batches
// ---------------------------------------------------------------------------
inline int g_grid = 0;   // BK_TEAMS

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

// breg: each thread holds the basis and D in registers (uniform values) and
// reads them there instead of from SLM
template <typename T, int nq, int batch, bool breg = false>
void bk3_cu(const std::size_t nelmt, const T* __restrict__ basis, const T* __restrict__ dbasis,
            const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int nm3 = nm * nm * nm;
    constexpr int block = round16(batch * nq2);
    static_assert(block <= 1024, "a team holds at most 1024 threads");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);
    const int nteams = (g_grid > 0 && g_grid < nbatches) ? g_grid : nbatches;

    #pragma omp target teams num_teams(nteams) thread_limit(block) \
        map(to: basis[:nm*nq], dbasis[:nq2], G[:nelmt*6*nq3], in[:nelmt*nm3]) map(from: out[:nelmt*nm3])
    {
        T s_basis[nm * nq];
        T s_dbasis[nq2];
        T s_wsp0[batch * nq3];
        T s_wsp1[batch * nq3];
        T s_rqr[batch * nq3];
        T s_rqs[batch * nq3];
        T* s_rqt = s_wsp0;
        #pragma omp parallel
        {
            T Bv[breg ? nm * nq : 1];
            T Dv[breg ? nq2 : 1];
            if constexpr (breg) {
                #pragma unroll
                for (int x = 0; x < nm * nq; ++x) {
                    Bv[x] = basis[x];
                }
                #pragma unroll
                for (int x = 0; x < nq2; ++x) {
                    Dv[x] = dbasis[x];
                }
            }
            const T* Bs = breg ? Bv : s_basis;
            const T* Ds = breg ? Dv : s_dbasis;
            #pragma omp for
            for (int t = 0; t < nm * nq; ++t) {
                s_basis[t] = basis[t];
            }
            #pragma omp for
            for (int t = 0; t < nq2; ++t) {
                s_dbasis[t] = dbasis[t];
            }
            for (int eb = omp_get_team_num(); eb < nbatches; eb += omp_get_num_teams()) {
                const int remaining = ne - eb * batch;
                const int cb = (batch < remaining) ? batch : remaining;
                const T* b_in = in + std::size_t(eb) * (batch * nm3);
                const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
                T* b_out = out + std::size_t(eb) * (batch * nm3);
                #pragma omp for
                for (int x = 0; x < cb * nm3; ++x) {
                    s_wsp0[x] = b_in[x];
                }
                // direction 0
                #pragma omp for
                for (int t = 0; t < cb * nm * nm; ++t) {
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
                            s += Bs[i * nq + p] * r[i];
                        }
                        s_wsp1[e * nq * nm * nm + (p * nm + j) * nm + k] = s;
                    }
                }
                // direction 1
                #pragma omp for
                for (int t = 0; t < cb * nq * nm; ++t) {
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
                            s += Bs[j * nq + q] * r[j];
                        }
                        s_wsp0[e * nq2 * nm + (q * nq + p) * nm + k] = s;
                    }
                }
                // direction 2: u(p,q,r) at wsp1(e,r,q,p)
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
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
                            s += Bs[k * nq + rr] * r[k];
                        }
                        s_wsp1[e * nq3 + (rr * nq + q) * nq + p] = s;
                    }
                }
                // derivatives, G (coalesced over r), fluxes
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    const auto [e, q, r] = split3<nq, nq>(t);
                    const T* u = s_wsp1 + e * nq3;
                    const T* eG = b_G + e * 6 * nq3 + q * nq + r;
                    T r_p[nq];
                    T r_q[nq];
                    T r_r[nq];
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        r_p[n] = u[(r * nq + q) * nq + n];
                        r_q[n] = Ds[n * nq + q];
                        r_r[n] = Ds[n * nq + r];
                    }
                    #pragma unroll
                    for (int p = 0; p < nq; ++p) {
                        T qr = 0;
                        T qs = 0;
                        T qt = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            qr += Ds[n * nq + p] * r_p[n];
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
                // transposed derivatives -> wsp1(e,r,q,p)
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
                    const auto [e, q, r] = split3<nq, nq>(t);
                    const int eo = e * nq3;
                    T r_p[nq];
                    T r_q[nq];
                    T r_r[nq];
                    #pragma unroll
                    for (int n = 0; n < nq; ++n) {
                        r_p[n] = s_rqr[eo + (n * nq + q) * nq + r];
                        r_q[n] = Ds[q * nq + n];
                        r_r[n] = Ds[r * nq + n];
                    }
                    #pragma unroll
                    for (int p = 0; p < nq; ++p) {
                        T s = 0;
                        #pragma unroll
                        for (int n = 0; n < nq; ++n) {
                            s += r_p[n] * Ds[p * nq + n];
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
                // direction 2 back
                #pragma omp for
                for (int t = 0; t < cb * nq2; ++t) {
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
                            s += Bs[k * nq + rr] * r[rr];
                        }
                        s_wsp0[e * nm * nq2 + (k * nq + q) * nq + p] = s;
                    }
                }
                // direction 1 back
                #pragma omp for
                for (int t = 0; t < cb * nm * nq; ++t) {
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
                            s += Bs[j * nq + q] * r[q];
                        }
                        s_wsp1[e * nm * nm * nq + (k * nm + j) * nq + p] = s;
                    }
                }
                // direction 0 back
                #pragma omp for
                for (int t = 0; t < cb * nm * nm; ++t) {
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
                            s += Bs[i * nq + p] * r[p];
                        }
                        s_wsp0[e * nm3 + (i * nm + j) * nm + k] = s;
                    }
                }
                #pragma omp for
                for (int x = 0; x < cb * nm3; ++x) {
                    b_out[x] = s_wsp0[x];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sf: register-column sweeps through three padded SLM boxes
// ---------------------------------------------------------------------------
constexpr int c_stride(const int nq) { return (nq % 2 == 0) ? nq + 1 : nq; }

constexpr int sf_batch(const int nq, const int want, const int width)
{
    const int by_slm = 61440 / (3 * nq * nq * c_stride(nq) * 4);
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

constexpr int sf_block(const int nq, const int batch, const int width)
{
    const int w = round16(width * batch * nq * nq);
    return (w < 1024) ? w : 1024;
}

// breg: each thread holds B and D in registers (uniform values) instead of
// reading them from global memory in every step
template <typename T, int nq, int batch, int block, bool breg = false>
void bk3_sf(const std::size_t nelmt, const T* __restrict__ gB, const T* __restrict__ gD,
            const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int nq2 = nq * nq;
    constexpr int nq3 = nq2 * nq;
    constexpr int nm3 = nm * nm * nm;
    constexpr int pc = c_stride(nq);
    constexpr int ebox = nq2 * pc;
    static_assert(block <= 1024, "a team holds at most 1024 threads");
    static_assert(3 * batch * ebox * int(sizeof(T)) <= 61440, "three SLM boxes within 60 KB");
    const int nbatches = static_cast<int>((nelmt + batch - 1) / batch);
    const int ne = static_cast<int>(nelmt);
    // box index of (a, b, c) of element e
    const auto at = [](const int e, const int a, const int b, const int c) { return e * ebox + (a * nq + b) * pc + c; };

    #pragma omp target teams distribute thread_limit(block) \
        map(to: gB[:nm*nq], gD[:nq2], G[:nelmt*6*nq3], in[:nelmt*nm3]) map(from: out[:nelmt*nm3])
    for (int eb = 0; eb < nbatches; ++eb) {
        T lm[3 * batch * ebox];
        T* U = lm;
        T* R = lm + batch * ebox;
        T* S = lm + 2 * batch * ebox;
        const int remaining = ne - eb * batch;
        const int cb = (batch < remaining) ? batch : remaining;
        const T* b_in = in + std::size_t(eb) * (batch * nm3);
        const T* b_G = G + std::size_t(eb) * (batch * 6 * nq3);
        T* b_out = out + std::size_t(eb) * (batch * nm3);

        #pragma omp parallel
        {
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
            #pragma omp for
            for (int x = 0; x < cb * nm3; ++x) {
                const int e = x / nm3;
                const int y = x % nm3;
                S[at(e, y / (nm * nm), y / nm % nm, y % nm)] = b_in[x];
            }
            // 1: along k
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
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
            // 2: along j
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
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
            // 3: along i
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
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
                }
            }
            // A | B: derivatives along q and r
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
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
            // C: column along p, G, fluxes
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                const auto [e, q, r] = split3<nq, nq>(t);
                const T* eG = b_G + e * 6 * nq3 + q * nq + r;
                T col[nq];
                T fr[nq];
                #pragma unroll
                for (int n = 0; n < nq; ++n) {
                    col[n] = U[at(e, n, q, r)];
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
                    U[at(e, p, q, r)] = s;
                }
            }
            // E | F: transposed derivatives along q and r, in place
            #pragma omp for
            for (int t = 0; t < 2 * cb * nq2; ++t) {
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
            // 7: sum and back along p, in place in U's column
            #pragma omp for
            for (int t = 0; t < cb * nq2; ++t) {
                const auto [e, q, r] = split3<nq, nq>(t);
                T w[nq];
                #pragma unroll
                for (int p = 0; p < nq; ++p) {
                    const int x = at(e, p, q, r);
                    w[p] = U[x] + R[x] + S[x];
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
            // 8: back along q
            #pragma omp for
            for (int t = 0; t < cb * nm * nq; ++t) {
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
            // 9: back along r
            #pragma omp for
            for (int t = 0; t < cb * nm * nm; ++t) {
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
            #pragma omp for nowait
            for (int x = 0; x < cb * nm3; ++x) {
                const int e = x / nm3;
                const int y = x % nm3;
                b_out[x] = S[at(e, y / (nm * nm), y / nm % nm, y % nm)];
            }
        }
    }
}

using kernel_fn = void (*)(std::size_t, const float*, const float*, const float*, const float*, float*);

struct variant {
    const char* name;
    kernel_fn fn;
};

#define CU(B) {"cu_b" #B, &bk3_cu<float, NQ, cu_batch(NQ, B)>}
#define SF(B, W) {"sf_b" #B "x" #W, &bk3_sf<float, NQ, sf_batch(NQ, B, W), sf_block(NQ, sf_batch(NQ, B, W), W)>}
// suffix r: the basis and D held in registers (breg)
#define CUR(B) {"cu_b" #B "r", &bk3_cu<float, NQ, cu_batch(NQ, B), true>}
#define SFR(B, W) {"sf_b" #B "x" #W "r", &bk3_sf<float, NQ, sf_batch(NQ, B, W), sf_block(NQ, sf_batch(NQ, B, W), W), true>}

const variant variants[] = {
    {"ref", &bk::SumFactorization<float, NQ>},
    CU(1), CU(2), CU(4), CU(8), CU(16),
    SF(1, 1), SF(2, 1), SF(4, 1), SF(8, 1), SF(16, 1), SF(32, 1),
    SF(1, 2), SF(2, 2), SF(4, 2), SF(8, 2), SF(16, 2),
    CUR(1), CUR(2), CUR(4), SFR(1, 1), SFR(2, 1), SFR(1, 2), SFR(8, 2),
};

} // namespace

int main(int argc, char** argv)
{
    constexpr int nq = NQ;
    constexpr int nm = nq - 1;
    const std::size_t nelmt = (argc > 1) ? std::size_t(std::atoll(argv[1])) : std::size_t(1) << 17;
    const int ntests = (argc > 2) ? std::atoi(argv[2]) : 10;
    g_grid = bk::env_int("BK_TEAMS", 0);

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    std::vector<float> G(nelmt * 6 * nq * nq * nq);
    bk::fill_random_test_data(in, G);   // in in [-1, 1], G = 1.5 + [-1, 1]
    std::vector<double> ref(in.size());
    bk::reference_bk3(nq, nelmt, basis, D, G, in, ref);
    const double big = bk::max_abs(ref);
    std::vector<float> out(in.size());
    std::printf("BK3 offload: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d\n", nq - 2, nq, nelmt,
                in.size(), ntests);

    // the arrays stay on the device across the variants; each variant's
    // output is zeroed on the device and copied back after its runs
    const float* d_basis = basis.data();
    const float* d_D = D.data();
    const float* d_G = G.data();
    const float* d_in = in.data();
    float* d_out = out.data();
    const std::size_t n_in = in.size();
    const std::size_t n_G = G.size();
    const double bytes = sizeof(float) * (2.0 * n_in + n_G);
    #pragma omp target data map(to: d_basis[:nm*nq], d_D[:nq*nq], d_G[:n_G], d_in[:n_in]) map(alloc: d_out[:n_in])
    {
        for (const variant& v : variants) {
            if (!bk::selected(v.name)) {
                continue;
            }
            #pragma omp target teams distribute parallel for
            for (std::size_t i = 0; i < n_in; ++i) {
                d_out[i] = 0.0f;
            }
            v.fn(nelmt, d_basis, d_D, d_G, d_in, d_out);   // warm-up
            const double t = bk::time_min(ntests, [&] { v.fn(nelmt, d_basis, d_D, d_G, d_in, d_out); });
            #pragma omp target update from(d_out[:n_in])
            const double relerr = bk::rel_maxdiff(out, ref, big);
            std::printf("%-10s GDoF/s = %7.3f  GB/s = %6.1f  time = %8.3f ms  relerr = %.1e%s\n", v.name,
                        1e-9 * n_in / t, 1e-9 * bytes / t, 1e3 * t, relerr, bk::fail_mark(relerr));
        }
    }
    return 0;
}
