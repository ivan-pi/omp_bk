// BK1_amx.cpp -- BK1 mass-operator sum factorization on the Apple AMX units.
//
// Needs amx.h (instruction layer) and bk_common.h next to this file.
// Build (Apple Silicon):
//   clang++ -O2 -std=c++17 BK1_amx.cpp -o bk1_amx
//   clang++ -O2 -std=c++17 -fopenmp BK1_amx.cpp -o bk1_amx      (libomp)
//   clang++ -O2 -std=c++17 -DAMX_EMULATE BK1_amx.cpp ...        (emulated, for A/B checks)
// Build anywhere else (AMX emulated in software, for logic checks only):
//   g++ -O2 -std=c++17 BK1_amx.cpp -o bk1_amx_emu
//
// Run:  ./bk1_amx [p=2] [nelmt=524288] [ntests=5]
//   BK_RANDOM=1  use pseudo-random in/JxW instead of the constant 3.0/1.0
//                (constant data cannot detect index-transposition bugs).
//   BK_BATCH=E   elements per AMX batch, in lane chunks of 16 (default 16,
//                which is the measured optimum; forced to 16 by the dense
//                path and the SoA layout)
//   OMP_NUM_THREADS  default: number of cores minus one
//   BK_DENSE=0/1 dense element-matrix path (default: on for p <= 2)
//   BK_LAYOUT=soa  store in/JxW/out elements-on-lanes (chunks of 16) so the
//                kernel runs without any layout conversion
//   BK_NOREF=1   skip the serial reference comparison (throughput sweeps)
//   BK_NT=1      non-temporal stores of `out` (element-major layout, clang on
//                arm64 only; default off -- measured no effect on M2 Pro)
//
// Mapping onto AMX (fma32, matrix mode):   z[row][col] += y[row] * x[col]
//   X  = 16 consecutive ELEMENTS of one field entry (elements on the lanes)
//   Y  = one row of the coefficient table, zero-padded to 16 lanes
//   Z  = up to 4 independent 16x16 fp32 tiles, one per free-index combination
// Every load and store is a full aligned 64-byte line; the output index takes
// the contracted index's place, so no transposes or layout rotations occur:
//   (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r) [*JxW] -> (i,q,r) -> (i,j,r) -> (i,j,k)

#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstddef>

#include "bk_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "amx.h"     // AMX_* instruction macros: hardware on Apple Silicon, emulated elsewhere

namespace bk {
namespace amx {

using ::amx::op::ldxy;
using ::amx::op::stz;
using ::amx::op::fma;

// ---------------------------------------------------------------------------
// Elements-on-lanes layout.  A batch of E elements (padded to EP, a multiple
// of 16) stores every field with the element index FASTEST:
//     value(a0,a1,a2; e)  at  base + ((a0*A1 + a1)*A2 + a2)*EP + e
// A 64-byte load at (a0,a1,a2, e0) is 16 consecutive elements of one entry,
// so every ldx/stz is a full, aligned line and lanes never interact.
//
// One contraction over index position d (0, 1 or 2) of a 3-index field,
// replacing extent S by N in place:
//     out(.., n, ..; e) = sum_s in(.., s, ..; e) * coef[s][n]
// For each combination of the two free indices (one Z tile each, four at a
// time) and each 16-element chunk:
//     X = in(free, s; e0..e0+15)     (aligned load)
//     Y = coef[s]                    (loaded once per step, Y register s)
//     z[tile][n][e] (+)= y[n] * x[e]
// then rows n = 0..N-1 of the tile are stored to out(free, n; e0..e0+15).
// The output index simply takes the contracted index's place, so the index
// order (i,j,k) -> (p,q,r) -> (i,j,k) never changes and no transposes or
// rotations are needed.
// ---------------------------------------------------------------------------
template <int N>
inline void contract_dim(const int d, const int A0, const int A1, const int A2,
                         const float* __restrict__ in, float* __restrict__ out,
                         const float (*__restrict__ coef)[16], const int EP)
{
    static_assert(N >= 1 && N <= 16, "output extent must fit one Z tile");
    constexpr int TILES = 4;

    const int ext_in[3]  = {A0, A1, A2};
    int ext_out[3]       = {A0, A1, A2};
    ext_out[d] = N;
    const int S = ext_in[d];
    // strides (in floats / EP) of the two layouts
    const int st_in[3]  = {ext_in[1] * ext_in[2],  ext_in[2],  1};
    const int st_out[3] = {ext_out[1] * ext_out[2], ext_out[2], 1};
    const int u = (d == 0) ? 1 : 0, v = (d == 2) ? 1 : 2;     // the free index positions
    const int F = ext_in[u] * ext_in[v];

    // Coefficient rows: Y register s holds coef[s] for s < 7; s >= 7 share Y7
    // and are reloaded per round when S > 8.
    const int ny = std::min(S, 8);
    for (int s = 0; s < ny; ++s) AMX_LDY(ldxy(s, coef[s]));

    for (int e0 = 0; e0 < EP; e0 += 16) {
        for (int f0 = 0; f0 < F; f0 += TILES) {
            const int nt = std::min(TILES, F - f0);
            int base_in[TILES], base_out[TILES];
            for (int t = 0; t < nt; ++t) {
                const int f = f0 + t, fu = f / ext_in[v], fv = f % ext_in[v];
                base_in[t]  = (fu * st_in[u]  + fv * st_in[v])  * EP + e0;
                base_out[t] = (fu * st_out[u] + fv * st_out[v]) * EP + e0;
            }
            for (int s = 0; s < S; ++s) {
                int yreg = s;
                if (s >= 7 && S > 8) { yreg = 7; AMX_LDY(ldxy(7, coef[s])); }
                for (int t = 0; t < nt; ++t) AMX_LDX(ldxy(t, in + base_in[t] + s * st_in[d] * EP));
                for (int t = 0; t < nt; ++t) AMX_FMA32(fma(t, 64 * t, 64 * yreg, s == 0 ? 1 : 0));
            }
            for (int t = 0; t < nt; ++t)
                for (int n = 0; n < N; ++n)
                    AMX_STZ(stz(4 * n + t, out + base_out[t] + n * st_out[d] * EP));
        }
    }
}

// Coefficient tables, zero-padded to 16 lanes.
//   Bp[i][p] = B(i,p)   (rows i < nm, used by the forward steps)
//   BT[p][i] = B(i,p)   (rows p < nq, used by the reverse steps)
template <int nq>
struct Coef {
    static constexpr int nm = nq - 1;
    alignas(64) float Bp[nm][16];
    alignas(64) float BT[nq][16];
    explicit Coef(const float* basis) {
        std::memset(Bp, 0, sizeof Bp);
        std::memset(BT, 0, sizeof BT);
        for (int i = 0; i < nm; ++i)
            for (int p = 0; p < nq; ++p) {
                Bp[i][p] = basis[i * nq + p];
                BT[p][i] = basis[i * nq + p];
            }
    }
};

// ---------------------------------------------------------------------------
// Dense path (low order): the whole interpolation as one matrix
//     Bd(s, q) = B(i,p) B(j,q) B(k,r),   s = (i,j,k) in nm^3,  q = (p,q,r) in nq^3
// applied as an outer-product loop over s with elements on the lanes,
//     z[tile c][q'][e] += Bd(s, 16c+q') * in(s; e)
// so every tile is completely full.  Coefficients are stored [c][s][16] with s
// padded to a multiple of 4, and the batch is one lane chunk (EP = 16), so four
// consecutive s-planes of both operands are contiguous 256-byte blocks and are
// brought in with quad loads (M2+): per four planes, 1 ldx + C ldy + 4C fma.
// ---------------------------------------------------------------------------
template <int nq>
struct DenseCoef {
    static constexpr int nm = nq - 1, nm3 = nm * nm * nm, nq3 = nq * nq * nq;
    static constexpr int CF = (nq3 + 15) / 16, CR = (nm3 + 15) / 16;      // tiles forward / reverse
    static constexpr int SF = (nm3 + 3) & ~3,  SR = (nq3 + 3) & ~3;       // padded plane counts
    alignas(128) float Bf[CF][SF][16];    // Bf[c][s][q'] = Bd(s, 16c+q')
    alignas(128) float Br[CR][SR][16];    // Br[c][q][x'] = Bd(16c+x', q)
    explicit DenseCoef(const float* basis) {
        std::memset(Bf, 0, sizeof Bf);
        std::memset(Br, 0, sizeof Br);
        auto B = [&](int i, int p) { return basis[i * nq + p]; };
        for (int i = 0; i < nm; ++i) for (int j = 0; j < nm; ++j) for (int k = 0; k < nm; ++k)
            for (int p = 0; p < nq; ++p) for (int q = 0; q < nq; ++q) for (int r = 0; r < nq; ++r) {
                const int sidx = (i * nm + j) * nm + k, qidx = (p * nq + q) * nq + r;
                const float v = B(i, p) * B(j, q) * B(k, r);
                Bf[qidx / 16][sidx][qidx % 16] = v;
                Br[sidx / 16][qidx][sidx % 16] = v;
            }
    }
};

// out(16c + n'; e) = sum_{s<S} coef[c][s][n'] * in(s; e)      for c < C, lanes e < 16
// in : S planes of 16 floats, 128-byte aligned, readable up to plane SP-1 (SP = S rounded up to 4)
// out: N = 16C rows of 16 floats (rows >= the real extent are padding)
// Tiles are processed in rounds of four; each round re-streams the input planes.
template <int SP, int C>
inline void dense_step(const int S, const float* __restrict__ in, float* __restrict__ out,
                       const float (*__restrict__ coef)[SP][16], const int N)
{
    for (int cr = 0; cr < C; cr += 4) {
        const int nt = std::min(4, C - cr);
        for (int s0 = 0; s0 < S; s0 += 4) {
            AMX_LDX(ldxy(0, in + s0 * 16, false, true));             // X0..X3 = planes s0..s0+3
            // Y0..Y3 and Y4..Y7 hold the coefficient quads of two tiles at a time
            for (int c0 = 0; c0 < nt; c0 += 2) {
                const int nc = std::min(2, nt - c0);
                for (int c = 0; c < nc; ++c) AMX_LDY(ldxy(4 * c, &coef[cr + c0 + c][s0][0], false, true));
                for (int k = 0; k < 4 && s0 + k < S; ++k)
                    for (int c = 0; c < nc; ++c)
                        AMX_FMA32(fma(c0 + c, 64 * k, 64 * (4 * c + k), (s0 + k == 0) ? 1 : 0));
            }
        }
        for (int t = 0; t < nt; ++t)
            for (int n = 0; n < std::min(16, N - 16 * (cr + t)); ++n)
                AMX_STZ(stz(4 * n + t, out + (16 * (cr + t) + n) * 16));
    }
}

inline int round16(int e) { return (e + 15) & ~15; }

// ---------------------------------------------------------------------------
// Element-major <-> elements-on-lanes conversions (steps 1, 5, 9), done as
// 4x4 block transposes with 4-float vectors.  On arm64 clang lowers the
// shuffles to NEON zip/uzp; the same code builds on x86 for emulation runs.
// Remainders (n % 4 entries, Eb % 4 elements) are handled by scalar loops.
// The SoA layout skips these functions entirely.
// ---------------------------------------------------------------------------
typedef float v4f __attribute__((vector_size(16)));

inline v4f v4_load(const float* p)          { v4f v; std::memcpy(&v, p, 16); return v; }
inline void v4_store(float* p, v4f v)       { std::memcpy(p, &v, 16); }

// Streaming (non-temporal) 16-byte store for the `out` array: STNP on arm64
// with clang, a plain store elsewhere.  Off unless BK_NT=1 (no effect measured on M2 Pro).
typedef float v4f_u __attribute__((vector_size(16), aligned(4)));
#if defined(__aarch64__) && defined(__clang__) && __has_builtin(__builtin_nontemporal_store)
inline void v4_store_stream(float* p, v4f v) { __builtin_nontemporal_store((v4f_u)v, reinterpret_cast<v4f_u*>(p)); }
constexpr bool HAVE_NT_STORE = true;
#else
inline void v4_store_stream(float* p, v4f v) { v4_store(p, v); }
constexpr bool HAVE_NT_STORE = false;
#endif
inline bool use_nt_store() {
    static const bool on = [] { auto v = get_env("BK_NT"); return v ? std::atoi(v->c_str()) != 0 : false; }();
    return HAVE_NT_STORE && on;
}

// (a0 a1 a2 a3 | b0.. | c0.. | d0..) -> (a0 b0 c0 d0 | a1 b1 c1 d1 | ...)
inline void v4_transpose(v4f& r0, v4f& r1, v4f& r2, v4f& r3) {
    const v4f t0 = __builtin_shufflevector(r0, r1, 0, 4, 1, 5);
    const v4f t1 = __builtin_shufflevector(r0, r1, 2, 6, 3, 7);
    const v4f t2 = __builtin_shufflevector(r2, r3, 0, 4, 1, 5);
    const v4f t3 = __builtin_shufflevector(r2, r3, 2, 6, 3, 7);
    r0 = __builtin_shufflevector(t0, t2, 0, 1, 4, 5);
    r1 = __builtin_shufflevector(t0, t2, 2, 3, 6, 7);
    r2 = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
    r3 = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);
}

// dst(x; e) = src(e; x),  x < n, e < Eb, dst pitch EP
inline void to_lanes(const float* __restrict__ src, float* __restrict__ dst, int Eb, int EP, int n) {
    const int E4 = Eb & ~3, n4 = n & ~3;
    for (int e = 0; e < E4; e += 4) {
        const float* s0 = src + std::size_t(e) * n;
        for (int x = 0; x < n4; x += 4) {
            v4f r0 = v4_load(s0 + x), r1 = v4_load(s0 + n + x), r2 = v4_load(s0 + 2 * n + x), r3 = v4_load(s0 + 3 * n + x);
            v4_transpose(r0, r1, r2, r3);
            float* d = dst + std::size_t(x) * EP + e;
            v4_store(d, r0); v4_store(d + EP, r1); v4_store(d + 2 * EP, r2); v4_store(d + 3 * EP, r3);
        }
        for (int x = n4; x < n; ++x)
            for (int k = 0; k < 4; ++k) dst[std::size_t(x) * EP + e + k] = s0[std::size_t(k) * n + x];
    }
    for (int e = E4; e < Eb; ++e)
        for (int x = 0; x < n; ++x) dst[std::size_t(x) * EP + e] = src[std::size_t(e) * n + x];
}

// dst(e; x) = src(x; e)
inline void from_lanes(const float* __restrict__ src, float* __restrict__ dst, int Eb, int EP, int n) {
    const int E4 = Eb & ~3, n4 = n & ~3;
    const bool nt = use_nt_store();
    for (int e = 0; e < E4; e += 4) {
        float* d0 = dst + std::size_t(e) * n;
        for (int x = 0; x < n4; x += 4) {
            const float* sp = src + std::size_t(x) * EP + e;
            v4f r0 = v4_load(sp), r1 = v4_load(sp + EP), r2 = v4_load(sp + 2 * EP), r3 = v4_load(sp + 3 * EP);
            v4_transpose(r0, r1, r2, r3);
            if (nt) { v4_store_stream(d0 + x, r0); v4_store_stream(d0 + n + x, r1); v4_store_stream(d0 + 2 * n + x, r2); v4_store_stream(d0 + 3 * n + x, r3); }
            else    { v4_store(d0 + x, r0);        v4_store(d0 + n + x, r1);        v4_store(d0 + 2 * n + x, r2);        v4_store(d0 + 3 * n + x, r3); }
        }
        for (int x = n4; x < n; ++x)
            for (int k = 0; k < 4; ++k) d0[std::size_t(k) * n + x] = src[std::size_t(x) * EP + e + k];
    }
    for (int e = E4; e < Eb; ++e)
        for (int x = 0; x < n; ++x) dst[std::size_t(e) * n + x] = src[std::size_t(x) * EP + e];
}

// w(x; e) *= JxW(e; x)   (step-5 for the element-major layout)
inline void scale_lanes(float* __restrict__ w, const float* __restrict__ JxW, int Eb, int EP, int n) {
    const int E4 = Eb & ~3, n4 = n & ~3;
    for (int e = 0; e < E4; e += 4) {
        const float* j0 = JxW + std::size_t(e) * n;
        for (int x = 0; x < n4; x += 4) {
            v4f r0 = v4_load(j0 + x), r1 = v4_load(j0 + n + x), r2 = v4_load(j0 + 2 * n + x), r3 = v4_load(j0 + 3 * n + x);
            v4_transpose(r0, r1, r2, r3);
            float* d = w + std::size_t(x) * EP + e;
            v4_store(d,          v4_load(d)          * r0);
            v4_store(d + EP,     v4_load(d + EP)     * r1);
            v4_store(d + 2 * EP, v4_load(d + 2 * EP) * r2);
            v4_store(d + 3 * EP, v4_load(d + 3 * EP) * r3);
        }
        for (int x = n4; x < n; ++x)
            for (int k = 0; k < 4; ++k) w[std::size_t(x) * EP + e + k] *= j0[std::size_t(k) * n + x];
    }
    for (int e = E4; e < Eb; ++e)
        for (int x = 0; x < n; ++x) w[std::size_t(x) * EP + e] *= JxW[std::size_t(e) * n + x];
}

// One batch of Eb <= EP elements, sum-factorized path.
// soa = false: in_e/JxW_e/out_e are element-major (e; idx).
// soa = true : EP == 16 and in_e/JxW_e/out_e are already (idx; e) chunks; the
//              kernel then reads in_e and writes out_e directly.
// w0, w1: nq^3 * EP + 48 floats each, 128-byte aligned.
template <int nq>
inline void batch_sumfact(const bool soa, const int Eb, const int EP,
                          const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                          float* __restrict__ out_e, const Coef<nq>& C,
                          float* __restrict__ w0, float* __restrict__ w1)
{
    constexpr int nm = nq - 1, nm3 = nm * nm * nm, nq3 = nq * nq * nq;

    const float* src = in_e;
    if (!soa) { to_lanes(in_e, w0, Eb, EP, nm3); src = w0; }       // step-1

    // steps 2-4: (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r)
    contract_dim<nq>(0, nm, nm, nm, src, w1, C.Bp, EP);
    contract_dim<nq>(1, nq, nm, nm, w1, w0, C.Bp, EP);
    contract_dim<nq>(2, nq, nq, nm, w0, w1, C.Bp, EP);

    // step-5: quadrature weights
    if (soa) { for (int x = 0; x < nq3 * 16; ++x) w1[x] *= JxW_e[x]; }
    else      scale_lanes(w1, JxW_e, Eb, EP, nq3);

    // steps 6-8: (p,q,r) -> (i,q,r) -> (i,j,r) -> (i,j,k)
    contract_dim<nm>(0, nq, nq, nq, w1, w0, C.BT, EP);
    contract_dim<nm>(1, nm, nq, nq, w0, w1, C.BT, EP);
    contract_dim<nm>(2, nm, nm, nq, w1, soa ? out_e : w0, C.BT, EP);

    if (!soa) from_lanes(w0, out_e, Eb, EP, nm3);                   // step-9
}

// One batch of Eb <= 16 elements, dense path (EP == 16).
template <int nq>
inline void batch_dense(const bool soa, const int Eb,
                        const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                        float* __restrict__ out_e, const DenseCoef<nq>& D,
                        float* __restrict__ w0, float* __restrict__ w1)
{
    using DC = DenseCoef<nq>;
    constexpr int nm3 = DC::nm3, nq3 = DC::nq3;

    const float* src = in_e;
    if (!soa) { to_lanes(in_e, w0, Eb, 16, nm3); src = w0; }

    dense_step<DC::SF, DC::CF>(nm3, src, w1, D.Bf, nq3);           // (s; e) -> (q; e)

    if (soa) { for (int x = 0; x < nq3 * 16; ++x) w1[x] *= JxW_e[x]; }
    else      scale_lanes(w1, JxW_e, Eb, 16, nq3);

    dense_step<DC::SR, DC::CR>(nq3, w1, soa ? out_e : w0, D.Br, nm3);   // (q; e) -> (x; e)

    if (!soa) from_lanes(w0, out_e, Eb, 16, nm3);
}

struct Options {
    bool soa   = false;   // in/JxW/out already elements-on-lanes, chunks of 16
    bool dense = false;   // dense element-matrix path
    int  batch = 16;
};

template <int nq>
Options options(bool soa) {
    Options o;
    o.soa = soa;
    o.dense = (nq <= 4);
    if (auto v = get_env("BK_DENSE")) o.dense = std::atoi(v->c_str()) != 0;
    if (auto v = get_env("BK_BATCH")) o.batch = std::max(1, std::atoi(v->c_str()));
    if (o.dense || o.soa) o.batch = 16;
    return o;
}

// In SoA layout the arrays are chunks of 16 elements: (chunk, idx, lane), with
// nelmt rounded up to a whole chunk and, for the dense path's quad loads,
// 48 floats of readable padding after the last chunk.
inline std::size_t soa_size(std::size_t nelmt, std::size_t n) { return (nelmt + 15) / 16 * 16 * n + 48; }

template <int nq>
void SumFactorization(const std::size_t nelmt,
                      const float* __restrict__ basis,
                      const float* __restrict__ JxW,
                      const float* __restrict__ in,
                      float* __restrict__ out,
                      const Options& o)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nm3 = std::size_t(nm) * nm * nm;
    constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    const Coef<nq> C(basis);
    const DenseCoef<nq>* D = o.dense ? new DenseCoef<nq>(basis) : nullptr;
    const int E  = o.batch;
    const int EP = round16(E);
    const std::size_t nbatch = (nelmt + E - 1) / E;
    const std::size_t wsz = nq3 * EP + 48;             // +3 planes for quad-load overrun

    #pragma omp parallel
    {
        // AMX state is per thread: every thread must enable it itself.
        std::vector<float> storage(2 * wsz + 32, 0.f);            // +32 floats for 128-byte alignment
        float* w0 = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(storage.data()) + 127) & ~uintptr_t(127));
        float* w1 = w0 + wsz;
        AMX_SET();
        #pragma omp for schedule(dynamic,4)
        for (std::size_t b = 0; b < nbatch; ++b) {
            const std::size_t e0 = b * E;
            const int Eb = int(std::min<std::size_t>(E, nelmt - e0));
            const float* in_e  = in  + e0 * nm3;     // element-major and SoA (E == 16) agree
            const float* JxW_e = JxW + e0 * nq3;
            float*       out_e = out + e0 * nm3;
            if (o.dense) batch_dense<nq>(o.soa, Eb, in_e, JxW_e, out_e, *D, w0, w1);
            else         batch_sumfact<nq>(o.soa, Eb, EP, in_e, JxW_e, out_e, C, w0, w1);
        }
        AMX_CLR();
    }
    delete D;
}

} // namespace amx

// ---------------------------------------------------------------------------
// Serial reference (body of BK1.cpp without the target pragmas)
// ---------------------------------------------------------------------------
template <typename T, int nq, typename index_t = int>
void SumFactorizationRef(const std::size_t nelmt, const T* basis, const T* JxW,
                         const T* in, T* out)
{
    constexpr int nm = nq - 1;
    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;
    using nq_cview = ndview<const T, nq, nq, nq>;
    const ndview<const T, nm, nq> B{basis};

    for (std::size_t e = 0; e < nelmt; ++e) {
        T scratch[2 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + nq * nq * nq};
        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        const nq_cview e_JxW{JxW + e * nq_cview::size};

        for (index_t i = 0; i < nm; ++i) for (index_t j = 0; j < nm; ++j) for (index_t k = 0; k < nm; ++k)
            wsp0(i, j, k) = e_in(i, j, k);
        for (index_t p = 0; p < nq; ++p) for (index_t k = 0; k < nm; ++k) for (index_t j = 0; j < nm; ++j) {
            T tmp = 0; for (index_t i = 0; i < nm; ++i) tmp += wsp0(i, j, k) * B(i, p); wsp1(p, j, k) = tmp; }
        for (index_t q = 0; q < nq; ++q) for (index_t p = 0; p < nq; ++p) for (index_t k = 0; k < nm; ++k) {
            T tmp = 0; for (index_t j = 0; j < nm; ++j) tmp += wsp1(p, j, k) * B(j, q); wsp0(q, p, k) = tmp; }
        for (index_t r = 0; r < nq; ++r) for (index_t q = 0; q < nq; ++q) for (index_t p = 0; p < nq; ++p) {
            T tmp = 0; for (index_t k = 0; k < nm; ++k) tmp += wsp0(q, p, k) * B(k, r); wsp1(p, q, r) = tmp; }
        for (index_t r = 0; r < nq; ++r) for (index_t q = 0; q < nq; ++q) for (index_t p = 0; p < nq; ++p)
            wsp1(p, q, r) *= e_JxW(p, q, r);
        for (index_t k = 0; k < nm; ++k) for (index_t q = 0; q < nq; ++q) for (index_t p = 0; p < nq; ++p) {
            T tmp = 0; for (index_t r = 0; r < nq; ++r) tmp += wsp1(p, q, r) * B(k, r); wsp0(q, p, k) = tmp; }
        for (index_t j = 0; j < nm; ++j) for (index_t k = 0; k < nm; ++k) for (index_t p = 0; p < nq; ++p) {
            T tmp = 0; for (index_t q = 0; q < nq; ++q) tmp += wsp0(q, p, k) * B(j, q); wsp1(p, j, k) = tmp; }
        for (index_t i = 0; i < nm; ++i) for (index_t j = 0; j < nm; ++j) for (index_t k = 0; k < nm; ++k) {
            T tmp = 0; for (index_t p = 0; p < nq; ++p) tmp += wsp1(p, j, k) * B(i, p); wsp0(i, j, k) = tmp; }
        for (index_t i = 0; i < nm; ++i) for (index_t j = 0; j < nm; ++j) for (index_t k = 0; k < nm; ++k)
            e_out(i, j, k) = wsp0(i, j, k);
    }
}

} // namespace bk

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------
template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    constexpr int nm = nq - 1;
    constexpr std::size_t nm3 = std::size_t(nm) * nm * nm, nq3 = std::size_t(nq) * nq * nq;

    const bool soa = get_env("BK_LAYOUT").value_or("") == "soa";
    const bk::amx::Options opt = bk::amx::options<nq>(soa);

    const std::array<T, nm * nq> basis = make_test_basis<T, nm, nq>();
    std::vector<T> JxW(nelmt * nq3, T(1.0));
    std::vector<T> in (nelmt * nm3, T(3.0));
    std::vector<T> out(nelmt * nm3);
    std::vector<T> ref(get_env("BK_NOREF") ? 0 : nelmt * nm3);

    if (get_env("BK_RANDOM")) {          // deterministic LCG, values in [-1, 1]
        uint32_t s = 12345u;
        auto next = [&] { s = 1664525u * s + 1013904223u; return T(s >> 8) / T(1 << 23) - T(1); };
        for (auto& v : in)  v = next();
        for (auto& v : JxW) v = T(1.5) + next();
    }

    // Kernel-side arrays: element-major, or SoA chunks of 16 elements
    // (chunk, idx, lane).  In SoA mode the conversion happens once here,
    // outside the timed region, as it would in a code that stores its
    // element-local data this way.
    std::vector<T> in_k, JxW_k, out_k;
    auto to_soa = [&](const std::vector<T>& a, std::size_t n) {
        std::vector<T> r(bk::amx::soa_size(nelmt, n), T(0));
        for (std::size_t e = 0; e < nelmt; ++e)
            for (std::size_t x = 0; x < n; ++x) r[((e / 16) * n + x) * 16 + e % 16] = a[e * n + x];
        return r;
    };
    if (soa) { in_k = to_soa(in, nm3); JxW_k = to_soa(JxW, nq3); out_k.assign(bk::amx::soa_size(nelmt, nm3), T(0)); }
    const T* d_in  = soa ? in_k.data()  : in.data();
    const T* d_JxW = soa ? JxW_k.data() : JxW.data();
    T*       d_out = soa ? out_k.data() : out.data();

    const std::size_t size_inout = in.size();
    const std::size_t size_JxW   = JxW.size();

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    double elapsed = std::numeric_limits<double>::max();

    for (int t = 0; t < ntests; ++t) {
        auto start = high_resolution_clock::now();
        bk::amx::SumFactorization<nq>(nelmt, basis.data(), d_JxW, d_in, d_out, opt);
        auto stop = high_resolution_clock::now();
        duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }
    if (soa)
        for (std::size_t e = 0; e < nelmt; ++e)
            for (std::size_t x = 0; x < nm3; ++x) out[e * nm3 + x] = out_k[((e / 16) * nm3 + x) * 16 + e % 16];

    const auto dof_rate  = [&](double s) { return 1.0e-9 * size_inout / s; };
    const auto byte_rate = [&](double s) { return 1.0e-9 * sizeof(T) * (2 * size_inout + size_JxW) / s; };

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::cout << "SumFactorization[AMX" << (AMX_HW ? "" : ", emulated")
              << (opt.dense ? ", dense" : ", sumfact") << (soa ? ", soa" : "")
              << (bk::amx::use_nt_store() ? ", nt" : "")
              << ", batch = " << opt.batch << ", threads = " << nthreads << "] -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";
    std::cout << "norm = " << norm2(out.data(), out.size()) << "\n";

    // Verification against the serial reference (BK_NOREF=1 skips it, for sweeps)
    if (get_env("BK_NOREF")) return;
    SumFactorizationRef<T, nq>(nelmt, basis.data(), JxW.data(), in.data(), ref.data());
    double max_err = 0, max_ref = 0;
    for (std::size_t x = 0; x < size_inout; ++x) {
        max_err = std::max(max_err, double(std::fabs(out[x] - ref[x])));
        max_ref = std::max(max_ref, double(std::fabs(ref[x])));
    }
    std::cout << "ref norm = " << norm2(ref.data(), ref.size())
              << "  max |amx - ref| = " << max_err
              << "  relative = " << max_err / max_ref << "\n";
}

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288

int main(int argc, char** argv)
{
#ifdef _OPENMP
    // Default thread count: all cores but one (9 on an M2 Pro), unless the
    // user set OMP_NUM_THREADS.  Ten threads on ten cores was measured to be
    // slower and noisier than nine because the last guided chunks straggle.
    if (!get_env("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, omp_get_num_procs() - 1));
#endif
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;

    switch (p) {
        case 1: run_test<float,  3>(nelmt, ntests); break;
        case 2: run_test<float,  4>(nelmt, ntests); break;
        case 3: run_test<float,  5>(nelmt, ntests); break;
        case 4: run_test<float,  6>(nelmt, ntests); break;
        case 5: run_test<float,  7>(nelmt, ntests); break;
        case 6: run_test<float,  8>(nelmt, ntests); break;
        case 7: run_test<float,  9>(nelmt, ntests); break;
        case 8: run_test<float, 10>(nelmt, ntests); break;
        case 9: run_test<float, 11>(nelmt, ntests); break;
        case 10: run_test<float, 12>(nelmt, ntests); break;
        case 11: run_test<float, 13>(nelmt, ntests); break;
        case 12: run_test<float, 14>(nelmt, ntests); break;
        case 13: run_test<float, 15>(nelmt, ntests); break;
        case 14: run_test<float, 16>(nelmt, ntests); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p << " (supported: 1..8)\n";
            return 1;
    }
    return 0;
}
