// bk3_gpu_steps.h -- the steps of the BK3 GPU kernels, one work item per
// call (t = the flattened work-item index of the step), shared by the
// OpenMP (openmp/offload/bk3_offload.cpp) and SYCL (sycl/bk3_sycl.cpp)
// kernels.  Plain functions on pointers, inlined into the kernels; the
// pointers of one call never alias.
//
// Operator (openmp/bk3_sumfact.h): u = B B B in, (qr, qs, qt) = derivatives
// of u with D(n, p), fluxes with G(e, f, p, q, r) (factor-major), out =
// B^T B^T B^T of the transposed derivatives with D(p, n).

#ifndef BK3_GPU_STEPS_H
#define BK3_GPU_STEPS_H

#include "bk1_gpu_steps.h"
#include "bk3_gpu.h"

namespace bk3gpu {

// cu, the port of the CUDA kernel: the interpolation steps are BK1's
// (bk1gpu::gs_steps s2, s3, s4u and s6, s7, s8, u(p,q,r) at u(e,r,q,p));
// these are the steps in between
template <typename T, int nq>
struct cu_steps {
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;

    // derivatives, G (coalesced over r) and fluxes; thread (e, q, r) walks p
    static void fluxes(const int t, const T* __restrict__ D, const T* __restrict__ bG, const T* __restrict__ w,
                       T* __restrict__ rqr, T* __restrict__ rqs, T* __restrict__ rqt)
    {
        const auto [e, q, r] = split3<nq, nq>(t);
        const T* u = w + e * nq3;
        const T* eG = bG + e * 6 * nq3 + q * nq + r;
        T r_p[nq];
        T r_q[nq];
        T r_r[nq];
        #pragma unroll
        for (int n = 0; n < nq; ++n) {
            r_p[n] = u[(r * nq + q) * nq + n];
            r_q[n] = D[n * nq + q];
            r_r[n] = D[n * nq + r];
        }
        #pragma unroll
        for (int p = 0; p < nq; ++p) {
            T qr = 0;
            T qs = 0;
            T qt = 0;
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                qr += D[n * nq + p] * r_p[n];
                qs += r_q[n] * u[(r * nq + n) * nq + p];
                qt += r_r[n] * u[(n * nq + q) * nq + p];
            }
            T g[6];
            #pragma unroll
            for (int f = 0; f < 6; ++f) {
                g[f] = eG[f * nq3 + p * nq2];
            }
            const int x = e * nq3 + (p * nq + q) * nq + r;
            apply_g(g, qr, qs, qt, rqr[x], rqs[x], rqt[x]);
        }
    }

    // transposed derivatives summed into w(e, r, q, p); thread (e, q, r) walks p
    static void divergence(const int t, const T* __restrict__ D, const T* __restrict__ rqr,
                           const T* __restrict__ rqs, const T* __restrict__ rqt, T* __restrict__ w)
    {
        const auto [e, q, r] = split3<nq, nq>(t);
        const int eo = e * nq3;
        T r_p[nq];
        T r_q[nq];
        T r_r[nq];
        #pragma unroll
        for (int n = 0; n < nq; ++n) {
            r_p[n] = rqr[eo + (n * nq + q) * nq + r];
            r_q[n] = D[q * nq + n];
            r_r[n] = D[r * nq + n];
        }
        #pragma unroll
        for (int p = 0; p < nq; ++p) {
            T s = 0;
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                s += r_p[n] * D[p * nq + n];
            }
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                s += rqs[eo + (p * nq + n) * nq + r] * r_q[n];
            }
            #pragma unroll
            for (int n = 0; n < nq; ++n) {
                s += rqt[eo + (p * nq + q) * nq + n] * r_r[n];
            }
            w[eo + (r * nq + q) * nq + p] = s;
        }
    }
};

// sf: register-column sweeps through three SLM boxes U, R, S of
// box_layout(nq, padded).  Steps (thread: what it computes):
//   load  in -> S(i,j,k)
//   1  (i,j): R(i,j,r) = B along k        6  (p,r): R = D^T along q, in place
//   2  (i,r): S(i,q,r) = B along j           (p,q): S = D^T along r, in place
//   3  (q,r): U(p,q,r) = B along i        7  (q,r): U(i,q,r) = B^T (U + R + S) along p
//   4  (p,r): R = D along q               8  (i,r): R(i,j,r) = B^T along q
//      (p,q): S = D along r               9  (i,j): S(i,j,k) = B^T along r
//   5  (q,r): qr = D along p, fluxes with G (coalesced over (q,r)): rs -> R,
//      rt -> S in place, U(:,q,r) = D^T rr      store  S -> out
// carry (one work-item per (q, r) column): the column of u (3 -> 5) and of
// D^T rr (5 -> 7) stay in the caller's ucol instead of U.
template <typename T, int nq, bool padded>
struct sf_steps {
    static constexpr int nm = nq - 1;
    static constexpr int nq2 = nq * nq;
    static constexpr int nq3 = nq2 * nq;
    static constexpr int nm3 = nm * nm * nm;
    static constexpr box_strides L = box_layout(nq, padded);
    static constexpr int ebox = L.se;

    static constexpr int at(const int e, const int a, const int b, const int c)
    {
        return e * L.se + a * L.sa + b * L.sb + c;
    }

    static void load(const int x, const T* __restrict__ in, T* __restrict__ S)
    {
        const int e = x / nm3;
        const int y = x % nm3;
        S[at(e, y / (nm * nm), y / nm % nm, y % nm)] = in[x];
    }

    static void store(const int x, const T* __restrict__ S, T* __restrict__ out)
    {
        const int e = x / nm3;
        const int y = x % nm3;
        out[x] = S[at(e, y / (nm * nm), y / nm % nm, y % nm)];
    }

    static void s1(const int t, const T* __restrict__ B, const T* __restrict__ S, T* __restrict__ R)
    {
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

    static void s2(const int t, const T* __restrict__ B, const T* __restrict__ R, T* __restrict__ S)
    {
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

    template <bool carry>
    static void s3(const int t, const T* __restrict__ B, const T* __restrict__ S, T* __restrict__ U, T (&ucol)[nq])
    {
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
            if constexpr (carry) {
                ucol[p] = s;
            }
        }
    }

    // t < cb nq^2: R along q; the rest: S along r
    static void s4(const int t, const int cb, const T* __restrict__ D, const T* __restrict__ U, T* __restrict__ R,
                   T* __restrict__ S)
    {
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

    template <bool carry>
    static void s5(const int t, const T* __restrict__ D, const T* __restrict__ bG, T* __restrict__ U,
                   T* __restrict__ R, T* __restrict__ S, T (&ucol)[nq])
    {
        const auto [e, q, r] = split3<nq, nq>(t);
        const T* eG = bG + e * 6 * nq3 + q * nq + r;
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

    // t < cb nq^2: R along q; the rest: S along r
    static void s6(const int t, const int cb, const T* __restrict__ D, T* __restrict__ R, T* __restrict__ S)
    {
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

    template <bool carry>
    static void s7(const int t, const T* __restrict__ B, T* __restrict__ U, const T* __restrict__ R,
                   const T* __restrict__ S, const T (&ucol)[nq])
    {
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

    static void s8(const int t, const T* __restrict__ B, const T* __restrict__ U, T* __restrict__ R)
    {
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

    static void s9(const int t, const T* __restrict__ B, const T* __restrict__ R, T* __restrict__ S)
    {
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
};

} // namespace bk3gpu

#endif // BK3_GPU_STEPS_H
