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
//                (default 16, the measured optimum; the dense path, the NEON
//                kernel and the SoA layout always use 16)
//   OMP_NUM_THREADS  default: number of cores minus one
//   BK_DENSE=0/1 dense element-matrix path (default: on for p <= 2)
//   BK_KERNEL=neon  cross-element NEON kernel (GNU vector types) instead of
//                AMX: the same layout and contractions, 4-wide FMAs in the core
//   BK_KERNEL=ref   the element-at-a-time reference kernel (BK1.cpp's loops,
//                reduction innermost), OpenMP over elements
//   BK_KERNEL=refv  the same with loops interchanged so every inner loop is
//                unit-stride and vectorizes (the CPU-correct loop order)
//   BK_NEON_NB=1|2|4|6|8  output rows per NEON register block (default 4;
//                4 NB + 5 live registers, so 6 still fits the 32 NEON
//                registers and 8 spills)
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
using ::amx::op::zrow_f32;
using ::amx::op::ALU_MAC;
using ::amx::op::ALU_MUL;

constexpr int LANES    = 16;            // elements per 64-byte line
constexpr int QUAD_PAD = 3 * LANES;     // readable planes past the end for a quad load's overrun

template <class I> constexpr I ceil_div(I x, int m) { return (x + m - 1) / m; }
template <class I> constexpr I round_up(I x, int m) { return ceil_div(x, m) * m; }

// Extents of one order: nm = p + 1 modes, nq = p + 2 quadrature points per direction.
template <int nq>
struct Dims {
    static constexpr int nm = nq - 1, nm3 = nm * nm * nm, nq3 = nq * nq * nq;
};

// ---------------------------------------------------------------------------
// Elements-on-lanes layout.  A batch of E elements (padded to EP, a multiple
// of 16) stores every field with the element index FASTEST:
//     value(a0,a1,a2; e)  at  base + ((a0*A1 + a1)*A2 + a2)*EP + e
// A 64-byte load at (a0,a1,a2, e0) is 16 consecutive elements of one entry,
// so every ldx/stz is a full, aligned line and lanes never interact.
//
// One contraction over index position D (0, 1 or 2) of a 3-index field with
// extents (A0, A1, A2) replaces that extent by N in place:
//     out(.., n, ..; e) = sum_s in(.., s, ..; e) * coef[s][n]
// The output index simply takes the contracted index's place, so the index
// order (i,j,k) -> (p,q,r) -> (i,j,k) never changes and no transposes or
// rotations are needed.  StepShape holds the geometry of one such step for
// both backends (AMX outer products, NEON vector FMAs).
// ---------------------------------------------------------------------------
template <int N, int D, int A0, int A1, int A2>
struct StepShape {
    static constexpr int ext_in[3]  = {A0, A1, A2};
    static constexpr int ext_out[3] = {D == 0 ? N : A0, D == 1 ? N : A1, D == 2 ? N : A2};
    static constexpr int S = ext_in[D];                                          // contracted extent
    static constexpr int st_in[3]  = {ext_in[1] * ext_in[2],  ext_in[2],  1};   // strides in planes
    static constexpr int st_out[3] = {ext_out[1] * ext_out[2], ext_out[2], 1};
    static constexpr int u = (D == 0) ? 1 : 0, v = (D == 2) ? 1 : 2;            // the free index positions
    static constexpr int F = ext_in[u] * ext_in[v];                              // free-index combinations
    static constexpr int ST_S = st_in[D], ST_N = st_out[D];                      // strides of s and n
    // plane offsets of free-index combination f in the input and the output
    static constexpr int in_base(int f)  { return (f / ext_in[v]) * st_in[u]  + (f % ext_in[v]) * st_in[v]; }
    static constexpr int out_base(int f) { return (f / ext_in[v]) * st_out[u] + (f % ext_in[v]) * st_out[v]; }
};

// AMX: for each combination of the two free indices (one Z tile each, four at
// a time) and each 16-element chunk:
//     X = in(free, s; e0..e0+15)     (aligned load)
//     Y = coef[s]                    (Y register s; rows s >= 7 share Y7 when S > 8)
//     z[tile][n][e] (+)= y[n] * x[e]
// then rows n = 0..N-1 of the tile are stored to out(free, n; e0..e0+15).
template <int N, int D, int A0, int A1, int A2>
inline void contract_dim(const float* __restrict__ in, float* __restrict__ out,
                         const float (*__restrict__ coef)[16], const int EP)
{
    static_assert(N >= 1 && N <= 16, "output extent must fit one Z tile");
    using Sh = StepShape<N, D, A0, A1, A2>;
    constexpr int S = Sh::S, F = Sh::F, TILES = 4;

    // Coefficient rows resident in Y0..Y6 (Y7 too when they all fit); for
    // S > 8 the rows from 7 on cycle through Y7, reloaded per round.
    constexpr bool cycle_y7 = S > 8;
    constexpr int ny = cycle_y7 ? 7 : S;
    for (int s = 0; s < ny; ++s) {
        AMX_LDY(ldxy(s, coef[s]));
    }

    for (int e0 = 0; e0 < EP; e0 += LANES) {
        for (int f0 = 0; f0 < F; f0 += TILES) {
            const int nt = std::min(TILES, F - f0);
            for (int s = 0; s < S; ++s) {
                int yreg = s;
                if (cycle_y7 && s >= 7) {
                    yreg = 7;
                    AMX_LDY(ldxy(7, coef[s]));
                }
                for (int t = 0; t < nt; ++t) {
                    AMX_LDX(ldxy(t, in + (Sh::in_base(f0 + t) + s * Sh::ST_S) * EP + e0));
                }
                for (int t = 0; t < nt; ++t) {
                    AMX_FMA32(fma(t, 64 * t, 64 * yreg, s == 0 ? ALU_MUL : ALU_MAC));
                }
            }
            for (int t = 0; t < nt; ++t) {
                for (int n = 0; n < N; ++n) {
                    AMX_STZ(stz(zrow_f32(t, n), out + (Sh::out_base(f0 + t) + n * Sh::ST_N) * EP + e0));
                }
            }
        }
    }
}

// Coefficient tables, zero-padded to 16 lanes.
//   Bp[i][p] = B(i,p)   (rows i < nm, used by the forward steps)
//   BT[p][i] = B(i,p)   (rows p < nq, used by the reverse steps)
template <int nq>
struct Coef : Dims<nq> {
    using Dims<nq>::nm;
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
struct DenseCoef : Dims<nq> {
    using Dims<nq>::nm;
    using Dims<nq>::nm3;
    using Dims<nq>::nq3;
    static constexpr int CF = ceil_div(nq3, 16), CR = ceil_div(nm3, 16);   // tiles forward / reverse
    static constexpr int SF = round_up(nm3, 4),  SR = round_up(nq3, 4);    // padded plane counts
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
// out: N rows of 16 floats (rows >= N of the last tile are not stored)
// Tiles are processed in rounds of four; each round re-streams the input planes.
template <int S, int N, int C, int SP>
inline void dense_step(const float* __restrict__ in, float* __restrict__ out,
                       const float (&__restrict__ coef)[C][SP][16])
{
    static_assert(SP >= round_up(S, 4) && 16 * C >= N, "coefficient table does not cover the step");
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
                        AMX_FMA32(fma(c0 + c, 64 * k, 64 * (4 * c + k), (s0 + k == 0) ? ALU_MUL : ALU_MAC));
                    }
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            for (int n = 0; n < std::min(16, N - 16 * (cr + t)); ++n) {
                AMX_STZ(stz(zrow_f32(t, n), out + (16 * (cr + t) + n) * 16));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4-float vectors (GNU vector types): the lane conversions and the NEON kernel.
// On arm64 clang lowers the shuffles to NEON zip/uzp; the same code builds on
// x86 for emulation runs.
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

// Four vectors from / to the rows of a 4x4 block at p with row pitch `pitch`
// (pitch 4: one contiguous 16-float lane line).
inline void load4x4(const float* p, std::size_t pitch, v4f& r0, v4f& r1, v4f& r2, v4f& r3) {
    r0 = v4_load(p);
    r1 = v4_load(p + pitch);
    r2 = v4_load(p + 2 * pitch);
    r3 = v4_load(p + 3 * pitch);
}

inline void store4x4(float* p, std::size_t pitch, v4f r0, v4f r1, v4f r2, v4f r3) {
    v4_store(p, r0);
    v4_store(p + pitch, r1);
    v4_store(p + 2 * pitch, r2);
    v4_store(p + 3 * pitch, r3);
}

// The 4x4 block at p, transposed.
inline void load4x4T(const float* p, std::size_t pitch, v4f& r0, v4f& r1, v4f& r2, v4f& r3) {
    load4x4(p, pitch, r0, r1, r2, r3);
    v4_transpose(r0, r1, r2, r3);
}

// ---------------------------------------------------------------------------
// Element-major <-> elements-on-lanes conversions (steps 1, 5, 9), done as
// 4x4 block transposes.  Remainders (n % 4 entries, Eb % 4 elements) are
// handled by scalar loops.  The SoA layout skips these functions entirely.
// ---------------------------------------------------------------------------

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
            v4f j0, j1, j2, j3;
            load4x4T(JxW + std::size_t(e) * n + x, n, j0, j1, j2, j3);
            float* d = w + std::size_t(x) * EP + e;
            v4f w0, w1, w2, w3;
            load4x4(d, EP, w0, w1, w2, w3);
            store4x4(d, EP, w0 * j0, w1 * j1, w2 * j2, w3 * j3);
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
// coefficient, so NB <= 6 fits the 32 NEON registers and NB = 8 spills.
// This is the honest CPU baseline: the blocked-across-cells structure of
// deal.II / libCEED on the same layout.  Batches are one lane chunk (EP = 16).
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
        v4f x0, x1, x2, x3;
        load4x4(pin + std::size_t(s) * ST_S * LANES, 4, x0, x1, x2, x3);
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
        store4x4(pout + std::size_t(n0 + n) * ST_N * LANES, 4, acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
    }
}

// One contraction (the StepShape of contract_dim) on NEON, NB output rows per
// register block; a tail of N % NB rows is split 4 + rest so that no block
// exceeds the register budget.
template <int NB, int N, int D, int A0, int A1, int A2>
inline void neon_contract(const float* __restrict__ in, float* __restrict__ out,
                          const float (*__restrict__ coef)[16])
{
    using Sh = StepShape<N, D, A0, A1, A2>;
    constexpr int S = Sh::S, ST_S = Sh::ST_S, ST_N = Sh::ST_N;
    constexpr int R = N % NB;

    for (int f = 0; f < Sh::F; ++f) {
        const float* pin  = in  + std::size_t(Sh::in_base(f))  * LANES;
        float*       pout = out + std::size_t(Sh::out_base(f)) * LANES;
        for (int n0 = 0; n0 + NB <= N; n0 += NB) {
            neon_block<NB, S, ST_S, ST_N>(n0, pin, pout, coef);
        }
        if constexpr (R > 4) {
            neon_block<4, S, ST_S, ST_N>(N - R, pin, pout, coef);
            neon_block<R - 4, S, ST_S, ST_N>(N - R + 4, pin, pout, coef);
        } else if constexpr (R > 0) {
            neon_block<R, S, ST_S, ST_N>(N - R, pin, pout, coef);
        }
    }
}

// ---------------------------------------------------------------------------
// The sum-factorized chain, written once for both backends.  An Op performs
// one contraction:  Op::contract<N, D, A0, A1, A2>(in, out, coef, EP).
// ---------------------------------------------------------------------------
struct AmxOp {
    static constexpr bool uses_amx = true;
    template <int N, int D, int A0, int A1, int A2>
    static void contract(const float* in, float* out, const float (*coef)[16], int EP) {
        contract_dim<N, D, A0, A1, A2>(in, out, coef, EP);
    }
};

template <int NB>
struct NeonOp {
    static constexpr bool uses_amx = false;
    template <int N, int D, int A0, int A1, int A2>
    static void contract(const float* in, float* out, const float (*coef)[16], int /*EP == 16*/) {
        neon_contract<NB, N, D, A0, A1, A2>(in, out, coef);
    }
};

// One batch of Eb <= EP elements: layout handling shared by all paths.
// soa = false: in_e/JxW_e/out_e are element-major (e; idx).
// soa = true : EP == 16 and in_e/JxW_e/out_e are already (idx; e) chunks; the
//              kernel then reads in_e and writes out_e directly.
// w0, w1: workspace of nq^3 * EP floats plus quad-load padding, 128-byte aligned.
// fwd(src, dst): nm^3 planes (idx; e) -> nq^3 planes;  rev(src, dst): the reverse.
template <int nq, class Fwd, class Rev>
inline void run_batch(const bool soa, const int Eb, const int EP,
                      const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                      float* __restrict__ out_e, float* __restrict__ w0, float* __restrict__ w1,
                      Fwd fwd, Rev rev)
{
    constexpr int nm3 = Dims<nq>::nm3, nq3 = Dims<nq>::nq3;

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

// Steps 2-4 and 6-8 with the contraction Op:
//   (i,j,k) -> (p,j,k) -> (p,q,k) -> (p,q,r) [*JxW] -> (i,q,r) -> (i,j,r) -> (i,j,k)
template <int nq, class Op>
inline void sumfact_batch(const bool soa, const int Eb, const int EP,
                          const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                          float* __restrict__ out_e, const Coef<nq>& C,
                          float* __restrict__ w0, float* __restrict__ w1)
{
    constexpr int nm = nq - 1;
    run_batch<nq>(soa, Eb, EP, in_e, JxW_e, out_e, w0, w1,
        [&](const float* src, float* dst) {
            Op::template contract<nq, 0, nm, nm, nm>(src, dst, C.Bp, EP);
            Op::template contract<nq, 1, nq, nm, nm>(dst, w0,  C.Bp, EP);
            Op::template contract<nq, 2, nq, nq, nm>(w0,  dst, C.Bp, EP);
        },
        [&](const float* src, float* dst) {
            Op::template contract<nm, 0, nq, nq, nq>(src, w0,  C.BT, EP);
            Op::template contract<nm, 1, nm, nq, nq>(w0,  w1,  C.BT, EP);    // src (== w1) is consumed
            Op::template contract<nm, 2, nm, nm, nq>(w1,  dst, C.BT, EP);
        });
}

// The dense element matrix, forward then reverse (EP == 16).
template <int nq>
inline void dense_batch(const bool soa, const int Eb,
                        const float* __restrict__ in_e, const float* __restrict__ JxW_e,
                        float* __restrict__ out_e, const DenseCoef<nq>& D,
                        float* __restrict__ w0, float* __restrict__ w1)
{
    using DC = DenseCoef<nq>;
    run_batch<nq>(soa, Eb, LANES, in_e, JxW_e, out_e, w0, w1,
        [&](const float* src, float* dst) {     // (s; e) -> (q; e)
            // quad loads need 128-byte alignment; an SoA chunk that only has
            // 64 (odd nm^3, odd chunk) is staged through w0 first
            if (reinterpret_cast<uintptr_t>(src) & 127) {
                std::memcpy(w0, src, DC::nm3 * LANES * sizeof(float));
                src = w0;
            }
            dense_step<DC::nm3, DC::nq3>(src, dst, D.Bf);
        },
        [&](const float* src, float* dst) {     // (q; e) -> (x; e)
            dense_step<DC::nq3, DC::nm3>(src, dst, D.Br);
        });
}

enum class KernelKind { amx, neon, ref, refv };

struct Options {
    KernelKind kernel = KernelKind::amx;
    bool soa     = false;   // in/JxW/out already elements-on-lanes, chunks of 16
    bool dense   = false;   // dense element-matrix path (AMX only)
    int  batch   = LANES;   // elements per batch; always 16 for dense, neon or soa
    int  neon_nb = 4;       // NEON output rows per register block

    // Read BK_KERNEL / BK_LAYOUT / BK_DENSE / BK_BATCH / BK_NEON_NB;
    // dense_default applies when BK_DENSE is unset.
    static Options from_env(bool dense_default) {
        auto env_int = [](const char* name, int fallback) {
            auto v = get_env(name);
            return v ? std::atoi(v->c_str()) : fallback;
        };
        Options o;
        const std::string kernel = get_env("BK_KERNEL").value_or("amx");
        if (kernel == "neon") {
            o.kernel = KernelKind::neon;
        } else if (kernel == "ref") {
            o.kernel = KernelKind::ref;
        } else if (kernel == "refv") {
            o.kernel = KernelKind::refv;
        } else if (kernel != "amx") {
            std::cerr << "BK_KERNEL=" << kernel << ": expected amx, neon, ref or refv\n";
            std::exit(1);
        }
        o.soa   = get_env("BK_LAYOUT").value_or("") == "soa";
        o.dense = (o.kernel == KernelKind::amx) && env_int("BK_DENSE", dense_default) != 0;
        o.batch = round_up(std::max(1, env_int("BK_BATCH", LANES)), LANES);
        if (o.dense || o.kernel != KernelKind::amx || o.soa) {
            o.batch = LANES;
        }
        const int nb = env_int("BK_NEON_NB", 4);
        o.neon_nb = (nb == 8 || nb == 6 || nb == 4 || nb == 2 || nb == 1) ? nb : 4;
        return o;
    }
};

// In SoA layout the arrays are chunks of 16 elements: (chunk, idx, lane), with
// nelmt rounded up to a whole chunk and, for the dense path's quad loads,
// QUAD_PAD floats of readable padding after the last chunk.
inline std::size_t soa_size(std::size_t nelmt, std::size_t n) {
    return round_up(nelmt, LANES) * n + QUAD_PAD;
}
inline std::size_t soa_index(std::size_t e, std::size_t x, std::size_t n) {
    return ((e / LANES) * n + x) * LANES + e % LANES;
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
class Kernel : public Dims<nq> {
public:
    using Dims<nq>::nm3;
    using Dims<nq>::nq3;

    Kernel(const float* basis, const Options& o)
        : o_(o), C_(basis), D_(o.dense ? std::make_unique<const DenseCoef<nq>>(basis) : nullptr) {}

    void operator()(const std::size_t nelmt, const float* __restrict__ JxW,
                    const float* __restrict__ in, float* __restrict__ out) const
    {
        auto chain = [&](auto op) {        // the sum-factorized chain with one contraction Op
            using Op = decltype(op);
            run(Op::uses_amx, nelmt, JxW, in, out,
                [&](int Eb, const float* in_e, const float* JxW_e, float* out_e, float* w0, float* w1) {
                    sumfact_batch<nq, Op>(o_.soa, Eb, o_.batch, in_e, JxW_e, out_e, C_, w0, w1);
                });
        };
        if (o_.kernel == KernelKind::neon) {
            switch (o_.neon_nb) {
                case 8: chain(NeonOp<8>{}); break;
                case 6: chain(NeonOp<6>{}); break;
                case 2: chain(NeonOp<2>{}); break;
                case 1: chain(NeonOp<1>{}); break;
                default: chain(NeonOp<4>{}); break;
            }
        } else if (o_.dense) {
            run(true, nelmt, JxW, in, out,
                [&](int Eb, const float* in_e, const float* JxW_e, float* out_e, float* w0, float* w1) {
                    dense_batch<nq>(o_.soa, Eb, in_e, JxW_e, out_e, *D_, w0, w1);
                });
        } else {
            chain(AmxOp{});
        }
    }

private:
    Options o_;
    Coef<nq> C_;
    std::unique_ptr<const DenseCoef<nq>> D_;   // 55 MB at nq = 16, so on the heap

    // The batch loop: batch(Eb, in_e, JxW_e, out_e, w0, w1) for every batch.
    template <class Batch>
    void run(const bool use_amx, const std::size_t nelmt, const float* __restrict__ JxW,
             const float* __restrict__ in, float* __restrict__ out, Batch batch) const
    {
        const int EP = o_.batch;                                     // a multiple of LANES
        const std::size_t nbatch = ceil_div(nelmt, EP);
        // quad-load overrun, rounded up so that w1 is 128-byte aligned too
        const std::size_t wsz = round_up(nq3 * EP + QUAD_PAD, 32);

        #pragma omp parallel
        {
            float* w0 = workspace(2 * wsz);
            float* w1 = w0 + wsz;
            if (use_amx) {
                AMX_SET();                // every thread enables AMX for itself
            }
            #pragma omp for schedule(dynamic,4)
            for (std::size_t b = 0; b < nbatch; ++b) {
                const std::size_t e0 = b * EP;
                const int Eb = int(std::min<std::size_t>(EP, nelmt - e0));
                // element-major and SoA (EP == 16) agree on the batch offsets
                batch(Eb, in + e0 * nm3, JxW + e0 * nq3, out + e0 * nm3, w0, w1);
            }
            if (use_amx) {
                AMX_CLR();
            }
        }
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

    using bk::amx::KernelKind;
    const bk::amx::Options opt = bk::amx::Options::from_env(/*dense_default=*/ nq <= 4);
    const bool ref_kernel = (opt.kernel == KernelKind::ref || opt.kernel == KernelKind::refv);
    const bool soa   = opt.soa;
    const bool noref = get_env("BK_NOREF").has_value();
    if (ref_kernel && soa) {
        std::cerr << "BK_KERNEL=ref/refv is element-major only\n";
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
        if (opt.kernel == KernelKind::refv) {
            SumFactorizationRefV<T, nq>(nelmt, basis.data(), JxW.data(), in.data(), out.data());
        } else if (opt.kernel == KernelKind::ref) {
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
    const char* label = opt.kernel == KernelKind::refv ? "reference, loops interchanged"
                      : opt.kernel == KernelKind::ref  ? "reference"
                      : opt.kernel == KernelKind::neon ? "NEON"
                      : AMX_HW                         ? "AMX" : "AMX, emulated";
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
