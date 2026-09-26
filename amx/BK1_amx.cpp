// BK1_amx.cpp -- BK1 mass-operator sum factorization on the Apple AMX units,
// with a NEON (GNU vector) kernel on the same elements-on-lanes layout as the
// CPU baseline.
//
// Needs amx.h (instruction layer) next to this file and bk_common.h in the
// parent directory.  From the repository root:  make BK1_amx
// Build by hand (Apple Silicon):
//   clang++ -O2 -std=c++17 BK1_amx.cpp -o bk1_amx
//   clang++ -O2 -std=c++17 -fopenmp BK1_amx.cpp -o bk1_amx      (libomp)
//   clang++ -O2 -std=c++17 -DAMX_EMULATE BK1_amx.cpp ...        (emulated, for A/B checks)
// Build anywhere else (AMX emulated in software, for logic checks only):
//   g++ -O2 -std=c++17 BK1_amx.cpp -o bk1_amx_emu
//
// Run:  ./bk1_amx [p=2] [nelmt=524288] [ntests=5]
//   BK_RANDOM=1  use pseudo-random in/JxW instead of the constant 3.0/1.0
//                (constant data cannot detect index-transposition bugs).
//   BK_BATCH=E   elements per AMX batch, rounded up to a multiple of 16
//                (default 16, the measured optimum; the dense path and the
//                SoA layout always use 16)
//   OMP_NUM_THREADS  default: number of cores minus one
//   BK_DENSE=0/1 dense element-matrix path (default: on for p <= 2)
//   BK_KERNEL=neon  cross-element NEON kernel (GNU vector types) instead of
//                AMX: the same layout and contractions, 4-wide FMAs in the core
//   BK_KERNEL=ref   the element-at-a-time reference kernel (BK1.cpp's loops,
//                reduction innermost), OpenMP over elements
//   BK_KERNEL=refv  the same with loops interchanged so every inner loop is
//                unit-stride and vectorizes (the CPU-correct loop order)
//   BK_NEON_NB=1|2|4|6|8  output rows per NEON register block (default 4;
//                4 NB + 5 live registers, so up to 6 fits the 32 NEON registers)
//   BK_LAYOUT=soa  store in/JxW/out elements-on-lanes (chunks of 16) so the
//                kernel runs without any layout conversion
//   BK_NOREF=1   skip the serial reference comparison (throughput sweeps)
//
// Mapping onto AMX (fma32, matrix mode):   z[row][col] += y[row] * x[col]
//   X  = 16 consecutive ELEMENTS of one field entry (elements on the lanes)
//   Y  = one row of the coefficient table, zero-padded to 16 lanes
//   Z  = up to 4 independent 16x16 fp32 tiles, one per free-index combination
// Every load and store is a full aligned 64-byte line; the output index takes
// the contracted index's place, so no transposes or layout rotations occur:
//   (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r) [*JxW] -> (i,q,r) -> (i,j,r) -> (i,j,k)

#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <new>

#include "../bk_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "amx.h"     // AMX_* instruction macros: hardware on Apple Silicon, emulated elsewhere

namespace bk {
namespace amx {

using ::amx::op::ldxy;
using ::amx::op::stz;
using ::amx::op::fma;

constexpr int LANES = 16;                       // elements per 64-byte line

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
    for (int s = 0; s < ny; ++s) {
        AMX_LDY(ldxy(s, coef[s]));
    }

    for (int e0 = 0; e0 < EP; e0 += LANES) {
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
                if (s >= 7 && S > 8) {
                    yreg = 7;
                    AMX_LDY(ldxy(7, coef[s]));
                }
                for (int t = 0; t < nt; ++t) {
                    AMX_LDX(ldxy(t, in + base_in[t] + s * st_in[d] * EP));
                }
                for (int t = 0; t < nt; ++t) {
                    AMX_FMA32(fma(t, 64 * t, 64 * yreg, s == 0 ? 1 : 0));
                }
            }
            for (int t = 0; t < nt; ++t) {
                for (int n = 0; n < N; ++n) {
                    AMX_STZ(stz(4 * n + t, out + base_out[t] + n * st_out[d] * EP));
                }
            }
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
        for (int i = 0; i < nm; ++i) {
            for (int p = 0; p < nq; ++p) {
                Bp[i][p] = basis[i * nq + p];
                BT[p][i] = basis[i * nq + p];
            }
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
// Pair and quad loads need 128-byte aligned addresses (corsix/amx, ldst.md).
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
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    const int sidx = (i * nm + j) * nm + k;
                    for (int p = 0; p < nq; ++p) {
                        for (int q = 0; q < nq; ++q) {
                            for (int r = 0; r < nq; ++r) {
                                const int qidx = (p * nq + q) * nq + r;
                                const float v = B(i, p) * B(j, q) * B(k, r);
                                Bf[qidx / 16][sidx][qidx % 16] = v;
                                Br[sidx / 16][qidx][sidx % 16] = v;
                            }
                        }
                    }
                }
            }
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
                for (int c = 0; c < nc; ++c) {
                    AMX_LDY(ldxy(4 * c, &coef[cr + c0 + c][s0][0], false, true));
                }
                for (int k = 0; k < 4 && s0 + k < S; ++k) {
                    for (int c = 0; c < nc; ++c) {
                        AMX_FMA32(fma(c0 + c, 64 * k, 64 * (4 * c + k), (s0 + k == 0) ? 1 : 0));
                    }
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            for (int n = 0; n < std::min(16, N - 16 * (cr + t)); ++n) {
                AMX_STZ(stz(4 * n + t, out + (16 * (cr + t) + n) * 16));
            }
        }
    }
}

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// Element-major <-> elements-on-lanes conversions (steps 1, 5, 9), done as
// 4x4 block transposes with 4-float vectors.  On arm64 clang lowers the
// shuffles to NEON zip/uzp; the same code builds on x86 for emulation runs.
// Remainders (n % 4 entries, Eb % 4 elements) are handled by scalar loops.
// The SoA layout skips these functions entirely.
// ---------------------------------------------------------------------------
typedef float v4f __attribute__((vector_size(16)));

inline v4f v4_load(const float* p) {
    v4f v;
    std::memcpy(&v, p, 16);
    return v;
}

inline void v4_store(float* p, v4f v) { std::memcpy(p, &v, 16); }

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

// Load the 4x4 block at p (row pitch `pitch`), transposed.
inline void load4x4T(const float* p, std::size_t pitch, v4f& r0, v4f& r1, v4f& r2, v4f& r3) {
    r0 = v4_load(p);
    r1 = v4_load(p + pitch);
    r2 = v4_load(p + 2 * pitch);
    r3 = v4_load(p + 3 * pitch);
    v4_transpose(r0, r1, r2, r3);
}

// Store four vectors as the rows of the 4x4 block at p (row pitch `pitch`).
inline void store4x4(float* p, std::size_t pitch, v4f r0, v4f r1, v4f r2, v4f r3) {
    v4_store(p, r0);
    v4_store(p + pitch, r1);
    v4_store(p + 2 * pitch, r2);
    v4_store(p + 3 * pitch, r3);
}

// Walk the (e < Eb) x (x < n) index space in 4x4 blocks: block(e, x) for every
// full block, scalar(e, x) for every remaining single position.
template <class Block, class Scalar>
inline void for_blocks(int Eb, int n, Block block, Scalar scalar) {
    const int E4 = Eb & ~3, n4 = n & ~3;
    for (int e = 0; e < E4; e += 4) {
        for (int x = 0; x < n4; x += 4) {
            block(e, x);
        }
        for (int x = n4; x < n; ++x) {
            for (int k = 0; k < 4; ++k) {
                scalar(e + k, x);
            }
        }
    }
    for (int e = E4; e < Eb; ++e) {
        for (int x = 0; x < n; ++x) {
            scalar(e, x);
        }
    }
}

// dst(x; e) = src(e; x),  x < n, e < Eb, dst pitch EP
inline void to_lanes(const float* __restrict__ src, float* __restrict__ dst, int Eb, int EP, int n) {
    for_blocks(Eb, n,
        [&](int e, int x) {
            v4f r0, r1, r2, r3;
            load4x4T(src + std::size_t(e) * n + x, n, r0, r1, r2, r3);
            store4x4(dst + std::size_t(x) * EP + e, EP, r0, r1, r2, r3);
        },
        [&](int e, int x) {
            dst[std::size_t(x) * EP + e] = src[std::size_t(e) * n + x];
        });
}

// dst(e; x) = src(x; e)
inline void from_lanes(const float* __restrict__ src, float* __restrict__ dst, int Eb, int EP, int n) {
    for_blocks(Eb, n,
        [&](int e, int x) {
            v4f r0, r1, r2, r3;
            load4x4T(src + std::size_t(x) * EP + e, EP, r0, r1, r2, r3);
            store4x4(dst + std::size_t(e) * n + x, n, r0, r1, r2, r3);
        },
        [&](int e, int x) {
            dst[std::size_t(e) * n + x] = src[std::size_t(x) * EP + e];
        });
}

// w(x; e) *= JxW(e; x)   (step-5 for the element-major layout)
inline void scale_lanes(float* __restrict__ w, const float* __restrict__ JxW, int Eb, int EP, int n) {
    for_blocks(Eb, n,
        [&](int e, int x) {
            v4f r0, r1, r2, r3;
            load4x4T(JxW + std::size_t(e) * n + x, n, r0, r1, r2, r3);
            float* d = w + std::size_t(x) * EP + e;
            v4_store(d,          v4_load(d)          * r0);
            v4_store(d + EP,     v4_load(d + EP)     * r1);
            v4_store(d + 2 * EP, v4_load(d + 2 * EP) * r2);
            v4_store(d + 3 * EP, v4_load(d + 3 * EP) * r3);
        },
        [&](int e, int x) {
            w[std::size_t(x) * EP + e] *= JxW[std::size_t(e) * n + x];
        });
}

// ---------------------------------------------------------------------------
// NEON kernel (GNU vector types): the same elements-on-lanes layout and the
// same six contractions as contract_dim, but the 16 x N x S product for one
// free-index combination is done with 4-wide vector FMAs in the core instead
// of outer products on the unit -- NB x 4 accumulator registers (NB output
// rows x 4 vectors of 4 elements), 4 registers for the plane, one broadcast
// coefficient.  This is the honest CPU baseline: the blocked-across-cells
// structure of deal.II / libCEED on the same layout.  Batches are one lane
// chunk (EP = 16).
// ---------------------------------------------------------------------------
#if defined(__clang__)
#define BK_UNROLL _Pragma("clang loop unroll(full)")
#else
#define BK_UNROLL
#endif

// out(n0 + n; e) = sum_s in(s; e) * coef[s][n0 + n]  for n < NB, lanes e < 16;
// plane strides ST_S (input) and ST_N (output) in units of 16 floats.
template <int NB, int S, int ST_S, int ST_N>
inline void neon_block(const int n0, const float* __restrict__ pin, float* __restrict__ pout,
                       const float (*__restrict__ coef)[16])
{
    v4f acc[NB][4] = {};
    auto plane = [&](int s) {
        const float* x = pin + std::size_t(s) * ST_S * LANES;
        const v4f x0 = v4_load(x);
        const v4f x1 = v4_load(x + 4);
        const v4f x2 = v4_load(x + 8);
        const v4f x3 = v4_load(x + 12);
        for (int n = 0; n < NB; ++n) {
            const float c = coef[s][n0 + n];
            const v4f b = {c, c, c, c};
            acc[n][0] += x0 * b;
            acc[n][1] += x1 * b;
            acc[n][2] += x2 * b;
            acc[n][3] += x3 * b;
        }
    };
    // Full unroll of the plane loop pays for short loops (S <= 8: removes the
    // loop and fill/drain overhead, +15-30% at p <= 4 on M2 Pro); for longer
    // loops the straight-line bodies cost instruction footprint for no gain,
    // so those are unrolled by two only.
    if constexpr (S <= 8) {
        BK_UNROLL
        for (int s = 0; s < S; ++s) {
            plane(s);
        }
    } else {
        int s = 0;
        for (; s + 2 <= S; s += 2) {
            plane(s);
            plane(s + 1);
        }
        if (s < S) {
            plane(s);
        }
    }
    for (int n = 0; n < NB; ++n) {
        float* o = pout + std::size_t(n0 + n) * ST_N * LANES;
        v4_store(o, acc[n][0]);
        v4_store(o + 4, acc[n][1]);
        v4_store(o + 8, acc[n][2]);
        v4_store(o + 12, acc[n][3]);
    }
}

// BK_NEON_NB: output rows per register block (1, 2, 4, 6 or 8; default 4)
inline int neon_block_rows() {
    static const int nb = [] {
        auto v = get_env("BK_NEON_NB");
        const int x = v ? std::atoi(v->c_str()) : 4;
        return (x == 8 || x == 6 || x == 4 || x == 2 || x == 1) ? x : 4;
    }();
    return nb;
}

// out(.., n, ..; e) = sum_s in(.., s, ..; e) * coef[s][n], contracting index
// position D of a field with extents (A0, A1, A2), one 16-lane chunk; all
// extents and strides are compile-time constants, so the s-loop unrolls.
template <int N, int D, int A0, int A1, int A2>
inline void neon_contract(const float* __restrict__ in, float* __restrict__ out,
                          const float (*__restrict__ coef)[16])
{
    constexpr int ext_in[3]  = {A0, A1, A2};
    constexpr int ext_out[3] = {D == 0 ? N : A0, D == 1 ? N : A1, D == 2 ? N : A2};
    constexpr int S = ext_in[D];
    constexpr int st_in[3]  = {ext_in[1] * ext_in[2],  ext_in[2],  1};
    constexpr int st_out[3] = {ext_out[1] * ext_out[2], ext_out[2], 1};
    constexpr int u = (D == 0) ? 1 : 0, v = (D == 2) ? 1 : 2;     // the free index positions
    constexpr int F = ext_in[u] * ext_in[v];
    constexpr int ST_S = st_in[D], ST_N = st_out[D];

    const int NB = neon_block_rows();
    for (int f = 0; f < F; ++f) {
        const int fu = f / ext_in[v], fv = f % ext_in[v];
        const float* pin  = in  + std::size_t(fu * st_in[u]  + fv * st_in[v])  * LANES;
        float*       pout = out + std::size_t(fu * st_out[u] + fv * st_out[v]) * LANES;
        int n0 = 0;
        for (; n0 + NB <= N; n0 += NB) {
            switch (NB) {
                case 8: neon_block<8, S, ST_S, ST_N>(n0, pin, pout, coef); break;
                case 6: neon_block<6, S, ST_S, ST_N>(n0, pin, pout, coef); break;
                case 4: neon_block<4, S, ST_S, ST_N>(n0, pin, pout, coef); break;
                case 2: neon_block<2, S, ST_S, ST_N>(n0, pin, pout, coef); break;
                default: neon_block<1, S, ST_S, ST_N>(n0, pin, pout, coef); break;
            }
        }
        // tail: the largest blocks that still fit (only sizes N allows are instantiated)
        while (n0 < N) {
            const int r = N - n0;
            if constexpr (N >= 4) {
                if (r >= 4) {
                    neon_block<4, S, ST_S, ST_N>(n0, pin, pout, coef);
                    n0 += 4;
                    continue;
                }
            }
            if constexpr (N >= 3) {
                if (r == 3) {
                    neon_block<3, S, ST_S, ST_N>(n0, pin, pout, coef);
                    n0 += 3;
                    continue;
                }
            }
            if constexpr (N >= 2) {
                if (r == 2) {
                    neon_block<2, S, ST_S, ST_N>(n0, pin, pout, coef);
                    n0 += 2;
                    continue;
                }
            }
            neon_block<1, S, ST_S, ST_N>(n0, pin, pout, coef);
            n0 += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// One batch of Eb <= EP elements: layout handling shared by all three paths.
// soa = false: in_e/JxW_e/out_e are element-major (e; idx).
// soa = true : EP == 16 and in_e/JxW_e/out_e are already (idx; e) chunks; the
//              kernel then reads in_e and writes out_e directly.
// w0, w1: workspace of nq^3 * EP floats plus quad-load padding, 128-byte aligned.
// fwd(src, dst): nm^3 planes (idx; e) -> nq^3 planes;  rev(src, dst): the reverse.
// ---------------------------------------------------------------------------
template <int nq, class Fwd, class Rev>
inline void run_batch(const bool soa, const int Eb, const int EP,
                      const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                      float* __restrict__ out_e, float* __restrict__ w0, float* __restrict__ w1,
                      Fwd fwd, Rev rev)
{
    constexpr int nm = nq - 1, nm3 = nm * nm * nm, nq3 = nq * nq * nq;

    // step-1
    const float* src = in_e;
    if (!soa) {
        to_lanes(in_e, w0, Eb, EP, nm3);
        src = w0;
    }

    fwd(src, w1);                                                   // steps 2-4

    // step-5: quadrature weights
    if (soa) {
        for (int x = 0; x < nq3 * EP; ++x) {
            w1[x] *= JxW_e[x];
        }
    } else {
        scale_lanes(w1, JxW_e, Eb, EP, nq3);
    }

    rev(w1, soa ? out_e : w0);                                      // steps 6-8

    // step-9
    if (!soa) {
        from_lanes(w0, out_e, Eb, EP, nm3);
    }
}

struct Options {
    bool soa   = false;   // in/JxW/out already elements-on-lanes, chunks of 16
    bool dense = false;   // dense element-matrix path (AMX)
    bool neon  = false;   // NEON cross-element kernel instead of AMX
    int  batch = LANES;   // elements per batch; always 16 for dense, neon or soa

    // Read BK_LAYOUT / BK_DENSE / BK_KERNEL / BK_BATCH; dense_default applies
    // when BK_DENSE is unset.
    static Options from_env(bool dense_default) {
        Options o;
        o.soa   = get_env("BK_LAYOUT").value_or("") == "soa";
        o.dense = dense_default;
        if (auto v = get_env("BK_DENSE")) {
            o.dense = std::atoi(v->c_str()) != 0;
        }
        if (get_env("BK_KERNEL").value_or("amx") == "neon") {
            o.neon  = true;
            o.dense = false;
        }
        if (auto v = get_env("BK_BATCH")) {
            o.batch = round_up(std::max(1, std::atoi(v->c_str())), LANES);
        }
        if (o.dense || o.neon || o.soa) {
            o.batch = LANES;
        }
        return o;
    }
};

// In SoA layout the arrays are chunks of 16 elements: (chunk, idx, lane), with
// nelmt rounded up to a whole chunk and, for the dense path's quad loads,
// 48 floats of readable padding after the last chunk.
inline std::size_t soa_size(std::size_t nelmt, std::size_t n) {
    return (nelmt + 15) / 16 * 16 * n + 48;
}
inline std::size_t soa_index(std::size_t e, std::size_t x, std::size_t n) {
    return ((e / 16) * n + x) * 16 + e % 16;
}

// Per-thread scratch space, 128-byte aligned, kept across calls so the timed
// kernel does not allocate.  AMX state is per thread as well.
inline float* workspace(std::size_t nfloats) {
    struct Buf {
        float* p = nullptr;
        std::size_t n = 0;
        ~Buf() { ::operator delete(p, std::align_val_t(128)); }
    };
    static thread_local Buf buf;
    if (buf.n < nfloats) {
        ::operator delete(buf.p, std::align_val_t(128));
        buf.p = static_cast<float*>(::operator new(nfloats * sizeof(float), std::align_val_t(128)));
        buf.n = nfloats;
    }
    return buf.p;
}

// The BK1 operator for one order: coefficient tables built once, then applied
// to any number of elements by operator().
template <int nq>
class Kernel {
public:
    static constexpr int nm = nq - 1;
    static constexpr std::size_t nm3 = std::size_t(nm) * nm * nm;
    static constexpr std::size_t nq3 = std::size_t(nq) * nq * nq;

    Kernel(const float* basis, const Options& o)
        : o_(o), C_(basis), D_(o.dense ? std::make_unique<const DenseCoef<nq>>(basis) : nullptr) {}

    const Options& options() const { return o_; }

    void operator()(const std::size_t nelmt, const float* __restrict__ JxW,
                    const float* __restrict__ in, float* __restrict__ out) const
    {
        const int E  = o_.batch;
        const int EP = round_up(E, LANES);
        const std::size_t nbatch = (nelmt + E - 1) / E;
        // + 3 planes for the quad-load overrun, rounded up to keep w1 128-byte aligned
        const std::size_t wsz = round_up(int(nq3 * EP + 48), 32);

        #pragma omp parallel
        {
            float* w0 = workspace(2 * wsz);
            float* w1 = w0 + wsz;
            if (!o_.neon) {
                AMX_SET();                // every thread enables AMX for itself
            }
            #pragma omp for schedule(dynamic,4)
            for (std::size_t b = 0; b < nbatch; ++b) {
                const std::size_t e0 = b * E;
                const int Eb = int(std::min<std::size_t>(E, nelmt - e0));
                const float* in_e  = in  + e0 * nm3;     // element-major and SoA (E == 16) agree
                const float* JxW_e = JxW + e0 * nq3;
                float*       out_e = out + e0 * nm3;
                if (o_.neon) {
                    neon_batch(Eb, in_e, JxW_e, out_e, w0, w1);
                } else if (o_.dense) {
                    dense_batch(Eb, in_e, JxW_e, out_e, w0, w1);
                } else {
                    sumfact_batch(Eb, EP, in_e, JxW_e, out_e, w0, w1);
                }
            }
            if (!o_.neon) {
                AMX_CLR();
            }
        }
    }

private:
    Options o_;
    Coef<nq> C_;
    std::unique_ptr<const DenseCoef<nq>> D_;   // 55 MB at nq = 16, so on the heap

    void sumfact_batch(int Eb, int EP, const float* in_e, const float* JxW_e, float* out_e,
                       float* w0, float* w1) const
    {
        const Coef<nq>& C = C_;
        run_batch<nq>(o_.soa, Eb, EP, in_e, JxW_e, out_e, w0, w1,
            [&](const float* src, float* dst) {     // (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r)
                contract_dim<nq>(0, nm, nm, nm, src, dst, C.Bp, EP);
                contract_dim<nq>(1, nq, nm, nm, dst, w0,  C.Bp, EP);
                contract_dim<nq>(2, nq, nq, nm, w0,  dst, C.Bp, EP);
            },
            [&](const float* src, float* dst) {     // (p,q,r) -> (i,q,r) -> (i,j,r) -> (i,j,k)
                contract_dim<nm>(0, nq, nq, nq, src, w0,  C.BT, EP);
                contract_dim<nm>(1, nm, nq, nq, w0,  w1,  C.BT, EP);    // src (== w1) is consumed
                contract_dim<nm>(2, nm, nm, nq, w1,  dst, C.BT, EP);
            });
    }

    void neon_batch(int Eb, const float* in_e, const float* JxW_e, float* out_e,
                    float* w0, float* w1) const
    {
        const Coef<nq>& C = C_;
        run_batch<nq>(o_.soa, Eb, LANES, in_e, JxW_e, out_e, w0, w1,
            [&](const float* src, float* dst) {     // (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r)
                neon_contract<nq, 0, nm, nm, nm>(src, dst, C.Bp);
                neon_contract<nq, 1, nq, nm, nm>(dst, w0,  C.Bp);
                neon_contract<nq, 2, nq, nq, nm>(w0,  dst, C.Bp);
            },
            [&](const float* src, float* dst) {     // (p,q,r) -> (i,q,r) -> (i,j,r) -> (i,j,k)
                neon_contract<nm, 0, nq, nq, nq>(src, w0,  C.BT);
                neon_contract<nm, 1, nm, nq, nq>(w0,  w1,  C.BT);      // src (== w1) is consumed
                neon_contract<nm, 2, nm, nm, nq>(w1,  dst, C.BT);
            });
    }

    void dense_batch(int Eb, const float* in_e, const float* JxW_e, float* out_e,
                     float* w0, float* w1) const
    {
        using DC = DenseCoef<nq>;
        const DC& D = *D_;
        run_batch<nq>(o_.soa, Eb, LANES, in_e, JxW_e, out_e, w0, w1,
            [&](const float* src, float* dst) {     // (s; e) -> (q; e)
                // quad loads need 128-byte alignment; an SoA chunk that only has
                // 64 (odd nm^3, odd chunk) is staged through w0 first
                if (reinterpret_cast<uintptr_t>(src) & 127) {
                    std::memcpy(w0, src, nm3 * LANES * sizeof(float));
                    src = w0;
                }
                dense_step<DC::SF, DC::CF>(nm3, src, dst, D.Bf, nq3);
            },
            [&](const float* src, float* dst) {     // (q; e) -> (x; e)
                dense_step<DC::SR, DC::CR>(nq3, src, dst, D.Br, nm3);
            });
    }
};

} // namespace amx

// ---------------------------------------------------------------------------
// Reference kernel (body of BK1.cpp without the target pragmas), OpenMP over
// elements; it is also the check every other kernel is compared against.
// ---------------------------------------------------------------------------
template <typename T, int nq>
void SumFactorizationRef(const std::size_t nelmt, const T* basis, const T* JxW,
                         const T* in, T* out)
{
    constexpr int nm = nq - 1;
    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;
    using nq_cview = ndview<const T, nq, nq, nq>;
    const ndview<const T, nm, nq> B{basis};

    #pragma omp parallel for schedule(guided)
    for (std::size_t e = 0; e < nelmt; ++e) {
        T scratch[2 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + nq * nq * nq};
        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        const nq_cview e_JxW{JxW + e * nq_cview::size};

        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    wsp0(i, j, k) = e_in(i, j, k);
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int k = 0; k < nm; ++k) {
                for (int j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (int i = 0; i < nm; ++i) {
                        tmp += wsp0(i, j, k) * B(i, p);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }
        for (int q = 0; q < nq; ++q) {
            for (int p = 0; p < nq; ++p) {
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (int j = 0; j < nm; ++j) {
                        tmp += wsp1(p, j, k) * B(j, q);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }
        for (int r = 0; r < nq; ++r) {
            for (int q = 0; q < nq; ++q) {
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (int k = 0; k < nm; ++k) {
                        tmp += wsp0(q, p, k) * B(k, r);
                    }
                    wsp1(p, q, r) = tmp;
                }
            }
        }
        for (int r = 0; r < nq; ++r) {
            for (int q = 0; q < nq; ++q) {
                for (int p = 0; p < nq; ++p) {
                    wsp1(p, q, r) *= e_JxW(p, q, r);
                }
            }
        }
        for (int k = 0; k < nm; ++k) {
            for (int q = 0; q < nq; ++q) {
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (int r = 0; r < nq; ++r) {
                        tmp += wsp1(p, q, r) * B(k, r);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }
        for (int j = 0; j < nm; ++j) {
            for (int k = 0; k < nm; ++k) {
                for (int p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (int q = 0; q < nq; ++q) {
                        tmp += wsp0(q, p, k) * B(j, q);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (int p = 0; p < nq; ++p) {
                        tmp += wsp1(p, j, k) * B(i, p);
                    }
                    wsp0(i, j, k) = tmp;
                }
            }
        }
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    e_out(i, j, k) = wsp0(i, j, k);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference with the loops interchanged for CPU SIMD: every contraction runs
// its reduction index OUTSIDE and a contiguous free (or output) index in the
// innermost loop, so the compiler can vectorize with unit-stride loads and no
// gathers -- the layout choice that the reference above lacks (there the
// reduction is innermost with stride-nq^2 loads, which stays scalar for
// p >= 2).  Same arithmetic, different summation order.
//   in(i,j,k) -i-> w1(p,j,k) -j-> w2(q,p,k) -k-> w3(p,q,r) [*JxW]
//             -r-> w4(q,p,k) -q-> w5(j,p,k) -p-> out(i,j,k)
// ---------------------------------------------------------------------------
template <typename T, int nq>
void SumFactorizationRefV(const std::size_t nelmt, const T* basis, const T* JxW,
                          const T* in, T* out)
{
    constexpr int nm = nq - 1, nm2 = nm * nm, nm3 = nm2 * nm, nq2 = nq * nq, nq3 = nq2 * nq;
    // B[i][p] (rows i) and BT[r][k] = B(k,r) (rows r), both row-contiguous
    T B[nm][nq], BT[nq][nm];
    for (int i = 0; i < nm; ++i) {
        for (int p = 0; p < nq; ++p) {
            B[i][p]  = basis[i * nq + p];
            BT[p][i] = basis[i * nq + p];
        }
    }

    #pragma omp parallel for schedule(guided)
    for (std::size_t e = 0; e < nelmt; ++e) {
        T w1[nq * nm2], w2[nq2 * nm], w3[nq3], w4[nq2 * nm], w5[nm * nq * nm];
        const T* e_in  = in  + e * nm3;
        const T* e_JxW = JxW + e * nq3;
        T*       e_out = out + e * nm3;

        // w1(p; j,k) = sum_i in(i; j,k) B(i,p)          inner: (j,k) contiguous, nm^2
        for (int p = 0; p < nq; ++p) {
            T* w = w1 + p * nm2;
            for (int x = 0; x < nm2; ++x) {
                w[x] = T(0);
            }
            for (int i = 0; i < nm; ++i) {
                const T b = B[i][p];
                const T* v = e_in + i * nm2;
                for (int x = 0; x < nm2; ++x) {
                    w[x] += v[x] * b;
                }
            }
        }
        // w2(q,p; k) = sum_j w1(p,j; k) B(j,q)          inner: k contiguous, nm
        for (int q = 0; q < nq; ++q) {
            for (int p = 0; p < nq; ++p) {
                T* w = w2 + (q * nq + p) * nm;
                for (int k = 0; k < nm; ++k) {
                    w[k] = T(0);
                }
                for (int j = 0; j < nm; ++j) {
                    const T b = B[j][q];
                    const T* v = w1 + (p * nm + j) * nm;
                    for (int k = 0; k < nm; ++k) {
                        w[k] += v[k] * b;
                    }
                }
            }
        }
        // w3(p,q; r) = sum_k w2(q,p; k) B(k,r)          inner: r contiguous (basis row), nq
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                T* w = w3 + (p * nq + q) * nq;
                for (int r = 0; r < nq; ++r) {
                    w[r] = T(0);
                }
                const T* v = w2 + (q * nq + p) * nm;
                for (int k = 0; k < nm; ++k) {
                    const T a = v[k];
                    for (int r = 0; r < nq; ++r) {
                        w[r] += a * B[k][r];
                    }
                }
            }
        }
        // quadrature weights, contiguous
        for (int x = 0; x < nq3; ++x) {
            w3[x] *= e_JxW[x];
        }
        // w4(q,p; k) = sum_r w3(p,q; r) B(k,r)          inner: k contiguous (BT row), nm
        for (int q = 0; q < nq; ++q) {
            for (int p = 0; p < nq; ++p) {
                T* w = w4 + (q * nq + p) * nm;
                for (int k = 0; k < nm; ++k) {
                    w[k] = T(0);
                }
                const T* v = w3 + (p * nq + q) * nq;
                for (int r = 0; r < nq; ++r) {
                    const T a = v[r];
                    for (int k = 0; k < nm; ++k) {
                        w[k] += a * BT[r][k];
                    }
                }
            }
        }
        // w5(j; p,k) = sum_q w4(q; p,k) B(j,q)          inner: (p,k) contiguous, nq*nm
        for (int j = 0; j < nm; ++j) {
            T* w = w5 + j * nq * nm;
            for (int x = 0; x < nq * nm; ++x) {
                w[x] = T(0);
            }
            for (int q = 0; q < nq; ++q) {
                const T b = B[j][q];
                const T* v = w4 + q * nq * nm;
                for (int x = 0; x < nq * nm; ++x) {
                    w[x] += v[x] * b;
                }
            }
        }
        // out(i,j; k) = sum_p w5(j,p; k) B(i,p)         inner: k contiguous, nm
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                T* w = e_out + (i * nm + j) * nm;
                for (int k = 0; k < nm; ++k) {
                    w[k] = T(0);
                }
                for (int p = 0; p < nq; ++p) {
                    const T b = B[i][p];
                    const T* v = w5 + (j * nq + p) * nm;
                    for (int k = 0; k < nm; ++k) {
                        w[k] += v[k] * b;
                    }
                }
            }
        }
    }
}

} // namespace bk

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------
template <int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    using T = float;                       // the AMX kernel is fp32 only
    constexpr int nm = nq - 1;
    constexpr std::size_t nm3 = std::size_t(nm) * nm * nm, nq3 = std::size_t(nq) * nq * nq;

    const bk::amx::Options opt = bk::amx::Options::from_env(/*dense_default=*/ nq <= 4);
    const std::string kernel_name = get_env("BK_KERNEL").value_or("amx");   // amx, neon, ref, refv
    const bool ref_kernel = (kernel_name == "ref" || kernel_name == "refv");
    const bool soa   = opt.soa;
    const bool noref = get_env("BK_NOREF").has_value();
    if (ref_kernel && soa) {
        std::cerr << "BK_KERNEL=" << kernel_name << " is element-major only\n";
        return;
    }

    const std::array<T, nm * nq> basis = make_test_basis<T, nm, nq>();
    std::vector<T> JxW(nelmt * nq3, T(1.0));
    std::vector<T> in (nelmt * nm3, T(3.0));
    std::vector<T> out(nelmt * nm3);

    if (get_env("BK_RANDOM")) {          // deterministic LCG, values in [-1, 1]
        uint32_t s = 12345u;
        auto next = [&] {
            s = 1664525u * s + 1013904223u;
            return T(s >> 8) / T(1 << 23) - T(1);
        };
        for (auto& v : in) {
            v = next();
        }
        for (auto& v : JxW) {
            v = T(1.5) + next();
        }
    }

    const std::size_t size_inout = in.size();
    const std::size_t size_JxW   = JxW.size();

    // Kernel-side arrays: element-major, or SoA chunks of 16 elements
    // (chunk, idx, lane).  In SoA mode the conversion happens once here,
    // outside the timed region, as it would in a code that stores its
    // element-local data this way.
    std::vector<T> in_k, JxW_k, out_k;
    auto to_soa = [&](const std::vector<T>& a, std::size_t n) {
        std::vector<T> r(bk::amx::soa_size(nelmt, n), T(0));
        for (std::size_t e = 0; e < nelmt; ++e) {
            for (std::size_t x = 0; x < n; ++x) {
                r[bk::amx::soa_index(e, x, n)] = a[e * n + x];
            }
        }
        return r;
    };
    if (soa) {
        in_k  = to_soa(in, nm3);
        JxW_k = to_soa(JxW, nq3);
        out_k.assign(bk::amx::soa_size(nelmt, nm3), T(0));
        if (noref) {                     // only the SoA copies are needed from here on
            std::vector<T>().swap(in);
            std::vector<T>().swap(JxW);
        }
    }
    const T* d_in  = soa ? in_k.data()  : in.data();
    const T* d_JxW = soa ? JxW_k.data() : JxW.data();
    T*       d_out = soa ? out_k.data() : out.data();

    // Coefficient tables are built once, outside the timed region, as in an application.
    const bk::amx::Kernel<nq> kernel(basis.data(), opt);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    double elapsed = std::numeric_limits<double>::max();

    for (int t = 0; t < ntests; ++t) {
        auto start = high_resolution_clock::now();
        if (kernel_name == "refv") {
            SumFactorizationRefV<T, nq>(nelmt, basis.data(), JxW.data(), in.data(), out.data());
        } else if (kernel_name == "ref") {
            SumFactorizationRef<T, nq>(nelmt, basis.data(), JxW.data(), in.data(), out.data());
        } else {
            kernel(nelmt, d_JxW, d_in, d_out);
        }
        auto stop = high_resolution_clock::now();
        duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }
    if (soa) {
        for (std::size_t e = 0; e < nelmt; ++e) {
            for (std::size_t x = 0; x < nm3; ++x) {
                out[e * nm3 + x] = out_k[bk::amx::soa_index(e, x, nm3)];
            }
        }
    }

    const auto dof_rate  = [&](double s) { return 1.0e-9 * size_inout / s; };
    const auto byte_rate = [&](double s) { return 1.0e-9 * sizeof(T) * (2 * size_inout + size_JxW) / s; };

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const char* label = kernel_name == "refv" ? "reference, loops interchanged"
                      : kernel_name == "ref"  ? "reference"
                      : opt.neon              ? "NEON"
                      : AMX_HW                ? "AMX" : "AMX, emulated";
    std::cout << "SumFactorization[" << label
              << (ref_kernel ? "" : opt.dense ? ", dense" : ", sumfact") << (soa ? ", soa" : "")
              << ", batch = " << opt.batch << ", threads = " << nthreads << "] -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";
    std::cout << "norm = " << norm2(out.data(), out.size()) << "\n";

    // Verification against the serial reference (BK_NOREF=1 skips it, for sweeps)
    if (noref) {
        return;
    }
    std::vector<T> ref(nelmt * nm3);
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
    // slower and noisier than nine because the last chunks straggle.
    if (!get_env("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() - 1));
    }
#endif
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;

    // Runtime p -> compile-time nq = p + 2; nq <= 16 so that one Z tile holds a row.
    switch (p) {
        case 1: run_test< 3>(nelmt, ntests); break;
        case 2: run_test< 4>(nelmt, ntests); break;
        case 3: run_test< 5>(nelmt, ntests); break;
        case 4: run_test< 6>(nelmt, ntests); break;
        case 5: run_test< 7>(nelmt, ntests); break;
        case 6: run_test< 8>(nelmt, ntests); break;
        case 7: run_test< 9>(nelmt, ntests); break;
        case 8: run_test<10>(nelmt, ntests); break;
        case 9: run_test<11>(nelmt, ntests); break;
        case 10: run_test<12>(nelmt, ntests); break;
        case 11: run_test<13>(nelmt, ntests); break;
        case 12: run_test<14>(nelmt, ntests); break;
        case 13: run_test<15>(nelmt, ntests); break;
        case 14: run_test<16>(nelmt, ntests); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p << " (supported: 1..14)\n";
            return 1;
    }
    return 0;
}
