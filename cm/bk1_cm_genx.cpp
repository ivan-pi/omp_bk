// bk1_cm_genx.cpp -- BK1 (mass operator) in C for Metal.  p <= 7: one element
// at a time per hardware thread, the whole element in registers; p = 8: a pair
// of threads per element, joined through SLM.  Compiled once per order with
// -DNQ=<nq>, nq = p + 2 (cm/build.sh).
//
// Orders p = 1..8.  The layout below is that of p = 1..6; p = 7, 8 (nq = 9,
// 10) store their tensors in split blocks instead, see "NQ > 8" further down.
//
// Layout in registers.  Every intermediate tensor is a matrix of rows x P
// floats, P = 4 (nq <= 4) or 8 (nq <= 8), the last index padded to P with
// zeros.  Each contraction sums over the slowest index and appends the new
// index as the fastest one (a rotation), so after three contractions the
// index order is back where it started:
//
//   in(i, j, k) -> u(j, k, p) -> v(k, p, q) -> x(p, q, r)        forward, B
//   x *= JxW(p, q, r)
//   x(p, q, r) -> y(q, r, i) -> z(r, i, j) -> o(i, j, k)          back, B^T
//
// Each contraction is a small GEMM out(m, n) = sum_s in(s, m) B(s, n): per s
// one vector mad over the whole output, the input replicated along n
// (region <1;P,0>) and the basis row replicated along m (region <0;P,1>),
// both legal Gen regions, so every instruction is a full SIMD16 mad.  The
// basis comes padded from the host: Bf(i, p) = B(i, p) and Bb(p, i) =
// B(i, p), zero beyond nq or nm columns, so the padding lanes stay zero.
//
// Memory: in and JxW by OWord block reads (DWord-aligned when an element's
// size is not a multiple of four floats), out by OWord block writes, or by
// scattered writes when it is not 16-byte aligned (odd nm).
//
// Kernels:
//   bk1_cm   the sum factorization, EPT consecutive elements per thread
//            (-DEPT=n, default 1), the loads of the next element issued
//            before the current one is computed (SIMD1 dispatch: the NDRange
//            counts threads = elements / EPT)
//   copy_cm  the copy bandwidth reference (cm_common.h)

#include <cm/cm.h>

#include "cm_common.h"

#ifndef NQ
#define NQ 6
#endif
#ifndef EPT
#define EPT 1            // elements per thread
#endif

constexpr int nq = NQ;
constexpr int nm = nq - 1;
constexpr int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : nq;   // padded fast index (exact beyond 8)
constexpr int nm3 = nm * nm * nm;
constexpr int nq3 = nq * nq * nq;

static_assert(nq >= 3 && nq <= 10, "p = 1..8");
static_assert(EPT >= 1, "elements per thread");

// --------------------------------------------------------------------------
// contraction: out(m, n) = sum_{s < S} in(s * STRIDE + m) * b(s, n), m < M, n < P
// --------------------------------------------------------------------------
template <int S, int M, int STRIDE, int NI, int NB, int NO>
_GENX_ inline void contract(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, NO> out)
{
    static_assert(NO >= M * P && NB >= S * P && NI >= (S - 1) * STRIDE + M, "region inside the operands");
    vector_ref<float, M * P> o = out.template select<M * P, 1>(0);
    o = in.template replicate<M, 1, P, 0>(0) * b.template replicate<M, 0, P, 1>(0);
#pragma unroll
    for (int s = 1; s < S; ++s) {
        o += in.template replicate<M, 1, P, 0>(s * STRIDE) * b.template replicate<M, 0, P, 1>(s * P);
    }
}

// --------------------------------------------------------------------------
// BK1
// --------------------------------------------------------------------------
constexpr bool in_aligned = (nm3 % 4 == 0);    // every element's in/out 16 B aligned
constexpr bool jxw_aligned = (nq3 % 4 == 0);
constexpr int NI = round4(nm3);
constexpr int NW = round4(nq3);
constexpr int NB = round4((nm + nq) * P);

#if NQ <= 8
// one element: u_in (in, packed), w (JxW, packed) -> od (out, packed)
_GENX_ inline void element(vector_ref<float, NI> u_in, vector_ref<float, NW> w, vector_ref<float, NB> b,
                           vector_ref<float, NI> od)
{
    vector_ref<float, nm * P> bf = b.select<nm * P, 1>(0);
    vector_ref<float, nq * P> bb = b.select<nq * P, 1>(nm * P);

    // forward: in(i, j, k) -> u(j, k, p) -> v(k, p, q) -> x(p, q, r)
    vector<float, nm * nm * P> u;
    contract<nm, nm * nm, nm * nm>(u_in, bf, u.select_all());
    vector<float, nm * P * P> v;
    contract<nm, nm * P, nm * P>(u.select_all(), bf, v.select_all());
    vector<float, nq * P * P> x;
    contract<nm, nq * P, P * P>(v.select_all(), bf, x.select_all());

    // weights: x(p, q, r) *= JxW(p, q, r) for p, q, r < nq (the rest is zero)
#pragma unroll
    for (int p = 0; p < nq; ++p) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
            x.select<nq, 1>((p * P + q) * P) *= w.select<nq, 1>((p * nq + q) * nq);
        }
    }

    // back: x(p, q, r) -> y(q, r, i) -> z(r, i, j) -> o(i, j, k)
    vector<float, nq * P * P> y;
    contract<nq, nq * P, P * P>(x.select_all(), bb, y.select_all());
    vector<float, nq * P * P> z;
    contract<nq, nq * P, P * P>(y.select_all(), bb, z.select_all());
    vector<float, nm * P * P> o;
    contract<nq, nm * P, P * P>(z.select_all(), bb, o.select_all());

    // o(i, j, k) for i, j, k < nm, packed
#pragma unroll
    for (int i = 0; i < nm; ++i) {
#pragma unroll
        for (int j = 0; j < nm; ++j) {
            od.select<nm, 1>((i * nm + j) * nm) = o.select<nm, 1>((i * P + j) * P);
        }
    }
}

_GENX_ inline void load_element(SurfaceIndex in, SurfaceIndex jxw, uint e, vector_ref<float, NI> u_in,
                                vector_ref<float, NW> w)
{
    load_blocks<in_aligned, 0, NI, NI>(in, e * (nm3 * 4), u_in);
    load_blocks<jxw_aligned, 0, NW, NW>(jxw, e * (nq3 * 4), w);
}
#endif

// out of element e, packed
_GENX_ inline void store_element(SurfaceIndex out, uint e, vector_ref<float, NI> od)
{
    if constexpr (in_aligned) {
        store_blocks<0, nm3, NI>(out, e * (nm3 * 4), od);
    } else {
        store_scattered<0, nm3, NI>(out, e * nm3, od);
    }
}

#if NQ <= 8

// EPT consecutive elements per thread; the loads of element t + 1 are issued
// before element t is computed
extern "C" _GENX_MAIN_ void bk1_cm(SurfaceIndex bpad [[type("buffer_t")]],
                                   SurfaceIndex jxw [[type("buffer_t")]],
                                   SurfaceIndex in [[type("buffer_t")]],
                                   SurfaceIndex out [[type("buffer_t")]],
                                   int nelmt)
{
    const uint e0 = (cm_group_id(0) * cm_local_size(0) + cm_local_id(0)) * EPT;
    if (e0 >= uint(nelmt)) {
        return;
    }

    // load: the first element's input and weights (the long latency), then
    // the padded basis Bf (nm x P) and Bb (nq x P), which stays in cache
    vector<float, NI> u_in;
    vector<float, NW> w;
    load_element(in, jxw, e0, u_in.select_all(), w.select_all());
    vector<float, NB> b;
    load_blocks<true, 0, NB, NB>(bpad, 0, b);

    const uint last = uint(nelmt) - 1;
#pragma unroll
    for (int t = 0; t < EPT; ++t) {
        const uint e = e0 + t;
        // prefetch the next element (past the end: the last one again)
        vector<float, NI> u_next;
        vector<float, NW> w_next;
        if (t + 1 < EPT) {
            load_element(in, jxw, (e + 1 < last) ? e + 1 : last, u_next.select_all(), w_next.select_all());
        }

        // compute and store
        vector<float, NI> od;
        element(u_in.select_all(), w.select_all(), b.select_all(), od.select_all());
        if (e < uint(nelmt)) {
            store_element(out, e, od.select_all());
        }
        if (t + 1 < EPT) {
            u_in = u_next;
            w = w_next;
        }
    }
}

#else  // NQ > 8: split tensors

// --------------------------------------------------------------------------
// nq = 9, 10 (p = 7, 8).  An extent of 9 or 10 is not a legal region width
// and padding it to 16 does not fit the register file, so every index that
// has been the fast (output) index of a contraction is stored split into a
// main part (8, or 4 for an extent of 5) and a tail (1 or 2).  A tensor is
// then a set of blocks, one per combination of parts of its split indices,
// each block row-major over its own extents and stored one after another in
// one register vector: no padding, the register footprint is the exact
// tensor size.  A contraction over the slowest index sums over its parts;
// per output block
//   main part 8 wide: one SIMD8 mad per row, the input a scalar broadcast and
//     the basis row contiguous (the replicated <1;8,0> operand of the padded
//     kernel is not a legal three-source operand and costs a mov);
//   narrower parts: one column at a time over all rows, SIMD16 mads into a
//     contiguous accumulator written once into the block.
// Every compile-time loop is a template recursion, so all offsets are
// constants (an #pragma unroll loop that is not unrolled turns them into
// indirect addressing).  The weights are loaded only after the forward
// contractions, so they are not live next to v and x.
//
// nq = 10 (p = 8): one element does not fit 256 GRFs (two ~1000-float
// tensors live at once), so a PAIR of threads shares it.  Thread t owns the
// quadrature index p in P_t = {4t..4t+3, 8+t} (a 4 + 1 split of extent 5):
// its forward pass computes u, v, x only for p in P_t (from a copy of Bf with
// the columns P_t first) and the weights of those p, and its backward pass
// sums over p in P_t only (rows P_t of Bb first), so each thread ends with a
// partial out.  Thread 1 hands its partial to thread 0 through shared local
// memory; thread 0 adds and stores.  JxW is read once; in is read by both.
// --------------------------------------------------------------------------
constexpr bool PAIR = (nq == 10);         // p = 8 (one thread per element spills)
constexpr int NPP = PAIR ? nq / 2 : nq;   // p values per thread

// main part of a split index of extent e, and the extent and first value of
// its part s
constexpr int pmain(const int e) { return (e >= 8) ? 8 : (e >= 4) ? 4 : e; }
constexpr int nparts(const int e, const bool split) { return (split && e > pmain(e)) ? 2 : 1; }
constexpr int pext(const int e, const bool split, const int s)
{
    return !split ? e : (s == 0) ? pmain(e) : e - pmain(e);
}
constexpr int pbase(const int e, const bool split, const int s) { return (split && s == 1) ? pmain(e) : 0; }
constexpr bool legal_width(const int d) { return d == 1 || d == 2 || d == 4 || d == 8; }

// tensor (E0, E1, E2), slow to fast; F: the index is stored split
template <int E0, int E1, int E2, bool F0, bool F1, bool F2>
struct tensor {
    static constexpr int e0 = E0, e1 = E1, e2 = E2;
    static constexpr bool f0 = F0, f1 = F1, f2 = F2;
    static constexpr int n0 = nparts(E0, F0), n1 = nparts(E1, F1), n2 = nparts(E2, F2);
    static constexpr int bsize(const int s0, const int s1, const int s2)
    {
        return pext(E0, F0, s0) * pext(E1, F1, s1) * pext(E2, F2, s2);
    }
    // offset of block (s0, s1, s2); blocks in s0-major order
    static constexpr int boff(const int s0, const int s1, const int s2)
    {
        int off = 0;
        for (int a = 0; a < n0; ++a) {
            for (int b = 0; b < n1; ++b) {
                for (int c = 0; c < n2; ++c) {
                    if (a == s0 && b == s1 && c == s2) {
                        return off;
                    }
                    off += bsize(a, b, c);
                }
            }
        }
        return off;
    }
    static constexpr int size = E0 * E1 * E2;
};

// o(m, 0:D) (+)= in(IB + m) * b(BB + 0:D) for rows m = R..M-1: one SIMD-D
// mad per row, the input a scalar broadcast and the basis row contiguous
// (legal three-source operands, no copies)
template <int D, int M, int IB, int BB, int OB, bool FIRST, int R, int NI, int NB, int NO>
_GENX_ inline void mad_rows(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, NO> o)
{
    if constexpr (R < M) {
        if constexpr (FIRST) {
            o.template select<D, 1>(OB + R * D) = in(IB + R) * b.template select<D, 1>(BB);
        } else {
            o.template select<D, 1>(OB + R * D) += in(IB + R) * b.template select<D, 1>(BB);
        }
        mad_rows<D, M, IB, BB, OB, FIRST, R + 1>(in, b, o);
    }
}

// the sum over a of a main-part block, a = A.. of part s0, then the next part
template <typename TI, typename TO, int BS, int s1, int s2, int sn, int s0, int A, int NI, int NB, int NO>
_GENX_ inline void contract_sum(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, NO> out)
{
    if constexpr (s0 < TI::n0) {
        if constexpr (A < pext(TI::e0, TI::f0, s0)) {
            constexpr int M = pext(TI::e1, TI::f1, s1) * pext(TI::e2, TI::f2, s2);   // rows (b, c)
            constexpr int D = pext(TO::e2, true, sn);                                // width of the n part
            constexpr int ag = pbase(TI::e0, TI::f0, s0) + A;                       // a in the basis
            constexpr int bb = ag * BS + pbase(TO::e2, true, sn);
            mad_rows<D, M, TI::boff(s0, s1, s2) + A * M, bb, TO::boff(s1, s2, sn), ag == 0, 0>(in, b, out);
            contract_sum<TI, TO, BS, s1, s2, sn, s0, A + 1>(in, b, out);
        } else {
            contract_sum<TI, TO, BS, s1, s2, sn, s0 + 1, 0>(in, b, out);
        }
    }
}

// one column t of a narrow block: acc(0:M) = sum_a in(rows of a) * B(a, t),
// a = A.. of part s0, then the next part (acc contiguous)
template <typename TI, typename TO, int BS, int s1, int s2, int sn, int t, int s0, int A, int M, int NI, int NB>
_GENX_ inline void column_sum(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, M> acc)
{
    if constexpr (s0 < TI::n0) {
        if constexpr (A < pext(TI::e0, TI::f0, s0)) {
            constexpr int ag = pbase(TI::e0, TI::f0, s0) + A;
            constexpr int ib = TI::boff(s0, s1, s2) + A * M;
            constexpr int bb = ag * BS + pbase(TO::e2, true, sn) + t;
            if constexpr (ag == 0) {
                acc = in.template select<M, 1>(ib) * b(bb);
            } else {
                acc += in.template select<M, 1>(ib) * b(bb);
            }
            column_sum<TI, TO, BS, s1, s2, sn, t, s0, A + 1>(in, b, acc);
        } else {
            column_sum<TI, TO, BS, s1, s2, sn, t, s0 + 1, 0>(in, b, acc);
        }
    }
}

// a narrow block (width D = 1, 2 or 4): each column summed contiguously over
// all a, then written once into the block (stride D)
template <typename TI, typename TO, int BS, int s1, int s2, int sn, int t, int NI, int NB, int NO>
_GENX_ inline void column_block(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, NO> out)
{
    constexpr int M = pext(TI::e1, TI::f1, s1) * pext(TI::e2, TI::f2, s2);
    constexpr int D = pext(TO::e2, true, sn);
    if constexpr (t < D) {
        vector<float, M> acc;
        column_sum<TI, TO, BS, s1, s2, sn, t, 0, 0>(in, b, acc.select_all());
        constexpr int ob = TO::boff(s1, s2, sn) + t;   // constexpr: folded, never a call
        out.template select<M, D>(ob) = acc;
        column_block<TI, TO, BS, s1, s2, sn, t + 1>(in, b, out);
    }
}

// every block of the output, K = linear block index
template <typename TI, typename TO, int BS, int K, int NI, int NB, int NO>
_GENX_ inline void contract_blocks(vector_ref<float, NI> in, vector_ref<float, NB> b, vector_ref<float, NO> out)
{
    if constexpr (K < TO::n0 * TO::n1 * TO::n2) {
        constexpr int s1 = K / (TO::n1 * TO::n2);
        constexpr int s2 = (K / TO::n2) % TO::n1;
        constexpr int sn = K % TO::n2;
        constexpr int D = pext(TO::e2, true, sn);
        static_assert(legal_width(D), "legal region width");
        if constexpr (D == 8) {
            contract_sum<TI, TO, BS, s1, s2, sn, 0, 0>(in, b, out);    // a mad per row
        } else {
            column_block<TI, TO, BS, s1, s2, sn, 0>(in, b, out);       // a mad per 16 rows
        }
        contract_blocks<TI, TO, BS, K + 1>(in, b, out);
    }
}

// out(b, c, n) = sum_a in(a, b, c) B(a, n), n < N, B rows of stride BS
template <typename TI, int N>
using contracted = tensor<TI::e1, TI::e2, N, TI::f1, TI::f2, true>;

template <typename TI, int N, int BS, int NI, int NB>
_GENX_ inline void contract_split(vector_ref<float, NI> in, vector_ref<float, NB> b,
                                  vector_ref<float, contracted<TI, N>::size> out)
{
    contract_blocks<TI, contracted<TI, N>, BS, 0>(in, b, out);
}

// The basis on the host (bpad, floats), P = nq:
//   0          Bf(i, p)  nm x nq, row stride P
//   nm * P     Bb(p, i)  nq x nm, row stride P
//   PAIR, from NB = round4((nm + nq) * P), for t = 0, 1:
//   NB + t * NBFT              Bf_t(i, c) = Bf(i, P_t[c])  nm x 5, row stride 8
//   NB + 2 * NBFT + t * NBBT   Bb_t(c, i) = Bb(P_t[c], i)  5 x nm, row stride P
constexpr int NBF = round4(nm * P);
constexpr int NBB = round4(nq * P);
constexpr int BST = 8;
constexpr int NBFT = round4(nm * BST);
constexpr int NBBT = round4(NPP * P);

using T_in = tensor<nm, nm, nm, false, false, false>;   // in(i, j, k), dense
using T_u = contracted<T_in, NPP>;                       // u(j, k, p), p in this thread's set
using T_v = contracted<T_u, nq>;                         // v(k, p, q)
using T_x = contracted<T_v, nq>;                         // x(p, q, r)
using T_y = contracted<T_x, nm>;                         // y(q, r, i) (PAIR: partial over p)
using T_z = contracted<T_y, nm>;                         // z(r, i, j)
using T_o = contracted<T_z, nm>;                         // o(i, j, k)

// x(p, q, r) *= JxW(p, q, r) for rows (p, q) = PQ..END-1 of this thread, each
// r part; w holds the weights of the rows from BASE on
template <int PQ, int END, int BASE, int NWC>
_GENX_ inline void apply_weights(vector_ref<float, T_x::size> x, vector_ref<float, NWC> w)
{
    if constexpr (PQ < END) {
        constexpr int p = PQ / nq;
        constexpr int q = PQ % nq;
        constexpr int sp = (p < pmain(NPP)) ? 0 : 1;
        constexpr int sq = (q < pmain(nq)) ? 0 : 1;
        constexpr int row = (p - pbase(NPP, true, sp)) * pext(nq, true, sq) + (q - pbase(nq, true, sq));
        constexpr int tr = nq - pmain(nq);
        constexpr int wo = (PQ - BASE) * nq;
        constexpr int x0 = T_x::boff(sp, sq, 0) + row * 8;    // constexpr: folded, never a call
        constexpr int x1 = T_x::boff(sp, sq, 1) + row * tr;
        x.template select<8, 1>(x0) *= w.template select<8, 1>(wo);
        x.template select<tr, 1>(x1) *= w.template select<tr, 1>(wo + 8);
        apply_weights<PQ + 1, END, BASE>(x, w);
    }
}

// load the weights of rows [R0, R1) of this thread's p (consecutive in
// memory from byte offset `off`) and apply them
template <int R0, int R1>
_GENX_ inline void weights(SurfaceIndex jxw, uint off, vector_ref<float, T_x::size> x)
{
    constexpr int n = (R1 - R0) * nq;
    vector<float, round4(n)> w;
    load_blocks<jxw_aligned, 0, round4(n), round4(n)>(jxw, off, w);
    apply_weights<R0, R1, R0>(x, w.select_all());
}

// od(i, j, k) = o(i, j, k), packed, for row (i, j) = IJ..
template <int IJ>
_GENX_ inline void pack_out(vector_ref<float, T_o::size> o, vector_ref<float, NI> od)
{
    if constexpr (IJ < nm * nm) {
        constexpr int i = IJ / nm;
        constexpr int j = IJ % nm;
        constexpr int si = (i < 8) ? 0 : 1;
        constexpr int sj = (j < 8) ? 0 : 1;
        constexpr int row = (i - si * 8) * pext(nm, true, sj) + (j - sj * 8);
        constexpr int tk = nm - 8;                 // tail of k (0 at p = 7)
        constexpr int tk1 = (tk > 0) ? tk : 1;   // a legal select size where unused
        constexpr int o0 = T_o::boff(si, sj, 0) + row * 8;    // constexpr: folded, never a call
        constexpr int o1 = T_o::boff(si, sj, 1) + row * tk;
        od.template select<8, 1>(IJ * nm) = o.template select<8, 1>(o0);
        if constexpr (tk > 0) {
            od.template select<tk1, 1>(IJ * nm + 8) = o.template select<tk1, 1>(o1);
        }
        pack_out<IJ + 1>(o, od);
    }
}

// PAIR: thread 1 hands its partial out to thread 0 in XR rounds of XC floats
// through one slot of XC floats per pair (SLM, not threads, would otherwise
// limit the threads per Xe-core); a barrier after each write and before the
// slot is reused.  Every thread of the group takes part in every barrier.
constexpr int MAX_PAIRS = 8;   // work-group of at most 16 threads (PAIR)
constexpr int XR = PAIR ? 3 : 1;   // handoff rounds (3: best at wg 4-8)
constexpr int XC = NI / XR;
static_assert(!PAIR || (NI % XR == 0 && XC % 4 == 0), "whole OWords per round");
constexpr int SLM_SLOT = XC * 4;   // bytes per pair

template <int R>
_GENX_ inline void handoff(uint slm, uint slot, uint t, vector_ref<float, NI> od)
{
    if constexpr (R < XR) {
        if (t == 1) {
            slm_blocks<true, R * XC, R * XC, XC, NI>(slm, slot, od);
        }
        cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
        cm_barrier();
        if (t == 0) {
            vector<float, NI> other;   // only the round's XC floats are used
            slm_blocks<false, R * XC, R * XC, XC, NI>(slm, slot, other.select_all());
            od.template select<XC, 1>(R * XC) += other.template select<XC, 1>(R * XC);
        }
        if constexpr (R + 1 < XR) {
            cm_barrier();   // thread 0 has read the slot before it is rewritten
        }
        handoff<R + 1>(slm, slot, t, od);
    }
}

extern "C" _GENX_MAIN_ void bk1_cm(SurfaceIndex bpad [[type("buffer_t")]],
                                   SurfaceIndex jxw [[type("buffer_t")]],
                                   SurfaceIndex in [[type("buffer_t")]],
                                   SurfaceIndex out [[type("buffer_t")]],
                                   int nelmt)
{
    static_assert(EPT == 1, "one element per thread (pair) at nq > 8 (registers)");
    const uint g = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint lid = cm_local_id(0);
    const uint t = PAIR ? (g & 1) : 0;   // PAIR: the thread's half of p
    const uint e = PAIR ? (g >> 1) : g;
    uint slm = 0;   // PAIR: the group's SLM buffer, this pair's slot at byte `slot`
    const uint slot = (lid >> 1) * SLM_SLOT;
    if constexpr (PAIR) {
        cm_slm_init(MAX_PAIRS * SLM_SLOT);
        slm = cm_slm_alloc(MAX_PAIRS * SLM_SLOT);
    } else {
        if (e >= uint(nelmt)) {
            return;
        }
    }
    // PAIR: every thread reaches the barrier; past the end compute element 0
    const uint ec = (e < uint(nelmt)) ? e : 0;

    // load: the input and the forward basis (PAIR: the columns P_t first)
    vector<float, NI> u_in;
    load_blocks<in_aligned, 0, NI, NI>(in, ec * (nm3 * 4), u_in);
    vector<float, NBF> bf;
    load_blocks<true, 0, NBF, NBF>(bpad, 0, bf);

    // forward: in(i, j, k) -> u(j, k, p) -> v(k, p, q) -> x(p, q, r)
    vector<float, T_u::size> u;
    if constexpr (PAIR) {
        vector<float, NBFT> bft;
        load_blocks<true, 0, NBFT, NBFT>(bpad, (NB + t * NBFT) * 4, bft);
        contract_split<T_in, NPP, BST>(u_in.select<T_in::size, 1>(0), bft.select_all(), u.select_all());
    } else {
        contract_split<T_in, NPP, P>(u_in.select<T_in::size, 1>(0), bf.select_all(), u.select_all());
    }
    vector<float, T_v::size> v;
    contract_split<T_u, nq, P>(u.select_all(), bf.select_all(), v.select_all());
    vector<float, T_x::size> x;
    contract_split<T_v, nq, P>(v.select_all(), bf.select_all(), x.select_all());

    // weights: loaded only now, so they are not live next to v and x
    // (PAIR: p = 4t..4t+3, then p = 8 + t)
    const uint wbase = ec * (nq3 * 4);
    if constexpr (PAIR) {
        weights<0, 4 * nq>(jxw, wbase + 4 * t * (nq * nq * 4), x.select_all());
        weights<4 * nq, NPP * nq>(jxw, wbase + (8 + t) * (nq * nq * 4), x.select_all());
    } else {
        weights<0, nq * nq>(jxw, wbase, x.select_all());
    }

    // back: x(p, q, r) -> y(q, r, i) -> z(r, i, j) -> o(i, j, k)
    vector<float, T_y::size> y;
    {
        constexpr int NBP = PAIR ? NBBT : NBB;
        vector<float, NBP> bp;   // rows of this thread's p
        if constexpr (PAIR) {
            load_blocks<true, 0, NBP, NBP>(bpad, (NB + 2 * NBFT + t * NBBT) * 4, bp);
        } else {
            load_blocks<(nm * P) % 4 == 0, 0, NBP, NBP>(bpad, nm * P * 4, bp);
        }
        contract_split<T_x, nm, P>(x.select_all(), bp.select_all(), y.select_all());
    }
    vector<float, NBB> bb;
    load_blocks<(nm * P) % 4 == 0, 0, NBB, NBB>(bpad, nm * P * 4, bb);
    vector<float, T_z::size> z;
    contract_split<T_y, nm, P>(y.select_all(), bb.select_all(), z.select_all());
    vector<float, T_o::size> o;
    contract_split<T_z, nm, P>(z.select_all(), bb.select_all(), o.select_all());

    // PAIR: thread 1 hands its partial out to thread 0 (handoff)
    vector<float, NI> od;
    pack_out<0>(o.select_all(), od.select_all());
    if constexpr (PAIR) {
        handoff<0>(slm, slot, t, od.select_all());
        if (t == 1 || e >= uint(nelmt)) {
            return;
        }
    }

    store_element(out, e, od.select_all());
}
#endif
