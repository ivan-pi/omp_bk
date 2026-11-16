// bk5_cm_genx.cpp -- BK5 (collocated Laplacian at the quadrature points) in C
// for Metal with the symmetric geometric factors (G6): one element per
// hardware thread, the element in registers, no shared local memory and no
// barriers.  Compiled once per order with -DNQ=<nq>, nq = p + 1 (cm/build.sh).
//
//   out = D_r^T (g_rr qr + g_rs qs + g_rt qt) + D_s^T (g_rs qr + g_ss qs + g_st qt)
//       + D_t^T (g_rt qr + g_st qs + g_tt qt),   (qr, qs, qt) = (D_r, D_s, D_t) in
//
// Layout in registers.  A tensor is stored as nq planes i, at a plane stride
// of S floats, in one of two forms:
//   flat    the nq x nq values (j, k) in memory order, so that a plane is
//           loaded by block reads straight from memory;
//   padded  nq rows j of P floats (P = 4 or 8 >= nq), each row aligned.
// Pointwise steps on whole planes (the r derivative and back-contraction)
// are SIMD over the plane.  The s and t steps and the fluxes work on rows:
// one P-wide scalar x vector mad per term, which reads rows of a flat plane
// as unaligned source regions (legal) and writes only aligned padded rows.
// (Writing rows into a flat plane makes the compiler realign every row with
// movs: ~2000 per element at nq = 7.)
//
// Per element (A: U, then O; B: Q, then R; g: the factors of one plane):
//   Q_i = sum_n D(i, n) U_n                                  r derivative, flat
//   for each plane i:  (the factors of plane i + 1 load meanwhile)
//     qs(j, :) = sum_n D(j, n) U_i(n, :)                     s derivative, padded
//     qt(j, :) = sum_n U_i(j, n) D(:, n)                     t derivative, padded
//     (rr, rs, rt) = G (qr, qs, qt) row by row; R_i = rr
//     O_i(j, :) = sum_n D(n, j) rs(n, :) + sum_n rt(j, n) D(n, :)
//   out_i = O_i + sum_n D(n, i) R_n                          r back, packed, stored
//
// Memory: the costly part, as compute is fully hidden (a build without any
// arithmetic runs at the same speed).  in is read plane by plane; G comes in
// layout 1 (G(e, i, j, f, k), bk::make_bk5_G), so the six factors of a plane
// are one contiguous run of 6 nq^2 floats (with factor-major G, six short
// unaligned runs per plane cost ~10% at odd nq).  out is packed into one flat
// element and written by OWord block writes of its 16 B aligned middle, with
// scattered writes for the head and tail at odd nq (four compile-time cases
// of e nq^3 mod 4).  Reads at odd nq are DWord aligned.  The basis comes
// padded from the host: Drow(n, k) = D(n, k), Dcol(n, k) = D(k, n), rows of P
// floats, zero beyond nq.
//
// p = 8 (nq = 9, P = 16) is compiled but does not fit the register file and
// spills (~8 KB): correct, but slow.
//
// Kernels: bk5_cm (EPT elements per thread, -DEPT=n, default 1, strided by
// the work-group size so that neighbouring threads work on neighbouring
// elements; the next element's first factors and in are loaded while the
// current one finishes; SIMD1 dispatch, the NDRange counts threads) and
// copy_cm (cm_common.h).

#include <cm/cm.h>

#include "cm_common.h"

#ifndef NQ
#define NQ 7
#endif
#ifndef EPT
#define EPT 1            // elements per thread
#endif

constexpr int nq = NQ;
constexpr int nq2 = nq * nq;
constexpr int nq3 = nq2 * nq;
constexpr int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : 16;   // row width of the s, t steps

static_assert(nq >= 2 && nq <= 9, "p = 1..8");
static_assert(EPT >= 1, "elements per thread");

constexpr int round8(const int n) { return (n + 7) / 8 * 8; }
constexpr int NP = round8(nq2 + P - nq);   // flat plane stride: the last row's P lanes fit
constexpr int NR = nq * P;                 // padded plane: nq rows of P
constexpr int S = (NP > NR) ? NP : NR;     // plane stride of U/O and Q/R (either form)
constexpr int NL = round4(nq2);            // floats read per plane (whole OWords)
constexpr int ND = round4(2 * nq * P);     // Drow, Dcol
constexpr int DCOL = nq * P;               // offset of Dcol
constexpr int NG = 6 * nq2;                // factors of one plane
constexpr int NGL = round4(NG);            // floats read per plane of G
constexpr int NGR = round8(NG + P - nq);   // ... and their register slot
constexpr bool aligned = (nq2 % 4 == 0);   // every plane 16 B aligned (even nq)
static_assert(NL <= NP && NGL <= NGR, "a read fits its slot");

// --------------------------------------------------------------------------
// plane steps
// --------------------------------------------------------------------------

// rows of a padded plane: o(j, :) = sum_n d(AJ j + AN n) x(n XS + :)     (HASX)
//                                  + sum_n y(j YS + n) d(BOFF + n P + :)  (HASY)
// x and y rows are P-wide source regions at any offset (XS, YS: row strides
// nq for a flat plane, P for a padded one); o rows are aligned, at j P
template <int AJ, int AN, int XS, bool HASX, int BOFF, int YS, bool HASY, int NX, int NY>
_GENX_ inline void rows(vector_ref<float, NX> x, vector_ref<float, NY> y, vector_ref<float, ND> d,
                        vector_ref<float, NR> o)
{
    static_assert(HASX || HASY, "at least one sum");
#pragma unroll
    for (int j = 0; j < nq; ++j) {
        vector_ref<float, P> r = o.select<P, 1>(j * P);
        if constexpr (HASX) {
            r = d(AJ * j) * x.select<P, 1>(0);
#pragma unroll
            for (int n = 1; n < nq; ++n) {
                r += d(AJ * j + AN * n) * x.select<P, 1>(n * XS);
            }
        }
        if constexpr (HASY) {
#pragma unroll
            for (int n = 0; n < nq; ++n) {
                if (!HASX && n == 0) {
                    r = y(j * YS) * d.select<P, 1>(BOFF);
                } else {
                    r += y(j * YS + n) * d.select<P, 1>(BOFF + n * P);
                }
            }
        }
    }
}

// the factors of plane i of element e, one contiguous run of NG floats
// (layout 1, G(e, i, j, f, k)): factor f of row j at g(j 6 nq + f nq)
_GENX_ inline void load_g(SurfaceIndex G, const uint e, const int i, vector_ref<float, NGR> g)
{
    load_blocks<aligned, 0, NGL, NGR>(G, (e * 6 * nq3 + i * NG) * 4, g);
}

// plane i of in for element e into its slot of A
_GENX_ inline void load_in(SurfaceIndex in, const uint e, const int i, vector_ref<float, nq * S> A)
{
    load_blocks<aligned, 0, NL, NP>(in, (e * nq3 + i * nq2) * 4, A.select<NP, 1>(i * S));
}

// r back-contraction and store of element e: out_i = O_i + sum_n D(n, i) R_n.
// As soon as plane i is computed, A_i is free and plane i of the next
// element's in is loaded into it (NEXT).  Even nq: every plane is 16 B
// aligned and stored on its own by block writes.  Odd nq: the planes are
// packed into st at offset H, st(0) being the 16 B aligned float at
// e nq3 - H, and stored as block writes of the aligned middle and scattered
// writes of the head and tail.
template <int H>
_GENX_ inline void finish(SurfaceIndex in, SurfaceIndex out, const uint e, const uint en, const bool store,
                          const bool next, vector_ref<float, nq * S> A, vector_ref<float, nq * S> B, vector_ref<float, ND> d)
{
    constexpr int NS = aligned ? NP : round8(H + nq3);
    constexpr int B0 = (H == 0) ? 0 : 4;     // aligned middle [B0, B1) of st
    constexpr int B1 = (H + nq3) / 4 * 4;
    vector<float, NS> st;
#pragma unroll
    for (int i = 0; i < nq; ++i) {
        vector<float, NR> o = A.select<NR, 1>(i * S);
#pragma unroll
        for (int n = 0; n < nq; ++n) {
            o += d(n * P + i) * B.select<NR, 1>(n * S);
        }
        if (next) {
            load_in(in, en, i, A);
        }
        const int OFF = aligned ? 0 : H + i * nq2;   // a constant after unrolling
#pragma unroll
        for (int j = 0; j < nq; ++j) {
            st.select<nq, 1>(OFF + j * nq) = o.select<nq, 1>(j * P);
        }
        if constexpr (aligned) {
            if (store) {
                store_blocks<0, nq2, NS>(out, (e * nq3 + i * nq2) * 4, st.select_all());
            }
        }
    }
    if constexpr (!aligned) {
        if (store) {
            const uint base = e * nq3 - H;
            store_scattered<H, B0 - H, NS>(out, base, st.select_all());
            store_blocks<B0, B1 - B0, NS>(out, base * 4, st.select_all());
            store_scattered<B1, H + nq3 - B1, NS>(out, base, st.select_all());
        }
    }
}

// --------------------------------------------------------------------------
// BK5
// --------------------------------------------------------------------------
extern "C" _GENX_MAIN_ void bk5_cm(SurfaceIndex dpad [[type("buffer_t")]],
                                   SurfaceIndex G [[type("buffer_t")]],
                                   SurfaceIndex in [[type("buffer_t")]],
                                   SurfaceIndex out [[type("buffer_t")]],
                                   int nelmt)
{
    // EPT elements per thread, strided by the work-group size W: at every
    // step the threads of a group work on W consecutive elements
    const uint W = cm_local_size(0);
    const uint e0 = cm_group_id(0) * EPT * W + cm_local_id(0);
    if (e0 >= uint(nelmt)) {
        return;
    }

    // load: in (flat planes) and the factors of plane 0 of the first
    // element, the padded basis (stays in cache).  A holds U, then O; B
    // holds Q, then R.
    vector<float, nq * S> A;
#pragma unroll
    for (int i = 0; i < nq; ++i) {
        load_in(in, e0, i, A.select_all());
    }
    vector<float, NGR> g;
    load_g(G, e0, 0, g.select_all());
    vector<float, ND> d;
    load_blocks<true, 0, ND, ND>(dpad, 0, d);
    vector<float, nq * S> B;

    // EPT elements; past nelmt the reads return 0 and nothing is stored
    for (int t = 0; t < EPT; ++t) {
        const uint e = e0 + t * W;
        const bool next = (t + 1 < EPT);

        // r derivative, flat planes: Q_i = sum_n D(i, n) U_n
#pragma unroll
        for (int i = 0; i < nq; ++i) {
            vector_ref<float, NP> q = B.select<NP, 1>(i * S);
            q = d(i * P) * A.select<NP, 1>(0);
#pragma unroll
            for (int n = 1; n < nq; ++n) {
                q += d(i * P + n) * A.select<NP, 1>(n * S);
            }
        }

#pragma unroll
        for (int i = 0; i < nq; ++i) {
            vector_ref<float, NP> u = A.select<NP, 1>(i * S);
            vector_ref<float, NP> qr = B.select<NP, 1>(i * S);

            // s and t derivatives of plane i, padded
            vector<float, NR> qs;
            vector<float, NR> qt;
            rows<P, 1, nq, true, 0, 0, false>(u, u, d, qs.select_all());
            rows<0, 0, 0, false, DCOL, nq, true>(u, u, d, qt.select_all());

            // fluxes, row by row: (rr, rs, rt) = G (qr, qs, qt); rt -> qt
            vector<float, NR> rr;
            vector<float, NR> rs;
#pragma unroll
            for (int j = 0; j < nq; ++j) {
                const int f = j * nq;       // row j of a flat plane
                const int c = j * P;        // row j of a padded plane
                const int gj = j * 6 * nq;  // row j of the factors
                vector<float, P> q0 = qr.select<P, 1>(f);
                vector_ref<float, P> q1 = qs.select<P, 1>(c);
                vector_ref<float, P> q2 = qt.select<P, 1>(c);
                rr.select<P, 1>(c) = g.select<P, 1>(gj) * q0 + g.select<P, 1>(gj + nq) * q1 +
                                     g.select<P, 1>(gj + 2 * nq) * q2;
                rs.select<P, 1>(c) = g.select<P, 1>(gj + nq) * q0 + g.select<P, 1>(gj + 3 * nq) * q1 +
                                     g.select<P, 1>(gj + 4 * nq) * q2;
                q2 = g.select<P, 1>(gj + 2 * nq) * q0 + g.select<P, 1>(gj + 4 * nq) * q1 +
                     g.select<P, 1>(gj + 5 * nq) * q2;
            }
            B.select<NR, 1>(i * S) = rr;

            // the next factors: plane i + 1, or plane 0 of the next element
            if (i + 1 < nq) {
                load_g(G, e, i + 1, g.select_all());
            } else if (next) {
                load_g(G, e + W, 0, g.select_all());
            }

            // s and t back-contractions into O_i (U_i is dead)
            rows<1, P, P, true, 0, P, true>(rs.select_all(), qt.select_all(), d, A.select<NR, 1>(i * S));
        }

        // r back-contraction and store (four cases of the alignment for odd
        // nq); the next element's in is loaded plane by plane meanwhile
        const bool store = (e < uint(nelmt));
        if constexpr (nq3 % 4 == 0) {
            finish<0>(in, out, e, e + W, store, next, A.select_all(), B.select_all(), d);
        } else {
            switch ((e * nq3) % 4) {
            case 0:
                finish<0>(in, out, e, e + W, store, next, A.select_all(), B.select_all(), d);
                break;
            case 1:
                finish<1>(in, out, e, e + W, store, next, A.select_all(), B.select_all(), d);
                break;
            case 2:
                finish<2>(in, out, e, e + W, store, next, A.select_all(), B.select_all(), d);
                break;
            default:
                finish<3>(in, out, e, e + W, store, next, A.select_all(), B.select_all(), d);
                break;
            }
        }
    }
}

