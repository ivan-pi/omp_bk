// BK1_ff.cpp -- BK1 (mass operator) in float-float arithmetic.
//
// SumFactorizationFF below has the interface of SumFactorization in double
// (bk1_sumfact.h): the caller passes IEEE doubles for the basis, the
// quadrature weights and the input and gets doubles back.  Inside the kernel
// every value is split into a pair of floats (hi + lo, ~48 significant bits)
// as it is loaded, the sum factorization runs on separate hi and lo work
// arrays with the helpers of float_float.h, and the result is joined back
// into a double as it is stored.  The point is hardware whose fp64 rate is a
// small fraction of fp32 (consumer GPUs): a float-float multiply-add costs
// 12 fp32 flops (plus 3 per column to renormalise), cheaper than fp64 at 1/32 or 1/64 of the fp32 rate, and the
// kernel reads and writes the same 8 bytes per value as double.
//
// Three float-float kernels share the arithmetic and differ in how the
// work is mapped: SumFactorizationFF runs one element per thread with the
// work arrays in thread-local memory; SumFactorizationFFTeams one element
// per team of ff_team_size(nq) threads, the work arrays shared by the team
// (shared memory on a GPU) and the points of each of the nine steps spread
// over the threads; SumFactorizationFFPlane a batch of elements per team of
// batch x nq x nq threads, each owning a column of its element's box in
// registers, with five phases and the contraction index as the outer loop;
// SumFactorizationFFPlaneLoop the same phases written with teams loop and
// nested loop constructs instead of a parallel region and worksharing
// loops, the form nvc++ lowers like OpenACC gang and vector loops.  Each
// plane kernel runs twice, with one element per team and with the batch of
// ff_plane_batch(nq) (about FF_PLANE_THREADS = 128 threads per team;
// -DFF_PLANE_BATCH=n forces a batch).  On the host the team kernels are
// slow, the first two open parallel regions per team and the loop one runs
// a team on one thread; they are meant for the GPU.
//
// The driver runs the double, float and the float-float kernels on the
// same data and prints one row per kernel (throughput and norm), then the
// relative L2 error of float and of each float-float kernel against the
// double result.  Norms and errors are reduced in native double.  The last
// line is the check: every float-float error must stay below the
// tolerance, else it reads FAIL and the exit status is 1.
// From the repository root:  make BK1_ff
//
// Run:  ./BK1_ff [p=2] [nelmt=524288] [ntests=5]
//   BK_RANDOM=1  pseudo-random in/JxW (bk_common.h) instead of the constant
//                3.0/1.0 of BK1; with the constant data the float row prints
//                BK1's norm.
//   BK_TOL=t     tolerance of the check (default 1e-10: random data gives
//                ~3e-15, the cancelling constant data at p = 1 ~1e-12, and a
//                broken float-float operation 1e-7 or worse).

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <string>
#include <type_traits>

#include "bk_common.h"
#include "float_float.h"
#include "bk1_sumfact.h"

using namespace bk;

namespace {

// ---------------------------------------------------------------------------
// The float-float kernel
// ---------------------------------------------------------------------------

// Flat indices: WSP into an nq^3 box (the work arrays and an element's JxW),
// DOF into an nm^3 block (an element's in and out), BAS into the nm x nq basis
// B(mode i, quad point p).  nq and nm are the enclosing template's constants.
#define WSP(i, j, k) (((i) * nq + (j)) * nq + (k))
#define DOF(i, j, k) (((i) * nm + (j)) * nm + (k))
#define BAS(i, p)    ((i) * nq + (p))

template <int nq, typename index_t = int>
void SumFactorizationFF(
    const std::size_t nelmt,
    const double* __restrict__ basis,
    const double* __restrict__ JxW,
    const double* __restrict__ in,
    double* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // The basis split into hi and lo, shared by all directions.
        float B_hi[nm * nq];
        float B_lo[nm * nq];
        for (index_t i = 0; i < nm * nq; ++i) {
            ff::split(basis[i], B_hi[i], B_lo[i]);
        }

        // Work arrays: two nq^3 boxes, each as a hi and a lo array; steps
        // address sub-slices of each box.  Every step assigns its full output
        // sub-slice, so no zeroing is needed.
        float wsp0_hi[nq3];
        float wsp0_lo[nq3];
        float wsp1_hi[nq3];
        float wsp1_lo[nq3];

        const double* e_in  = in  + e * nm3;
        double*       e_out = out + e * nm3;
        const double* e_JxW = JxW + e * nq3;

        // step-1 : split in -> wsp0 (nm^3 sub-block of the nq^3 box)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    ff::split(e_in[DOF(i, j, k)], wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)]);
                }
            }
        }

        // step-2 : direction 0
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t i = 0; i < nm; ++i) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, j, k)] = th;
                    wsp1_lo[WSP(p, j, k)] = tl;
                }
            }
        }

        // step-3 : direction 1
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t j = 0; j < nm; ++j) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(q, p, k)] = th;
                    wsp0_lo[WSP(q, p, k)] = tl;
                }
            }
        }

        // step-4 : direction 2
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t k = 0; k < nm; ++k) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, q, r)] = th;
                    wsp1_lo[WSP(p, q, r)] = tl;
                }
            }
        }

        // Reverse operations

        // step-5 : multiply with weights and determinant of Jacobi, each
        // JxW value split at its single use
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float jh;
                    float jl;
                    ff::split(e_JxW[WSP(p, q, r)], jh, jl);
                    ff::mul(wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)], jh, jl);
                }
            }
        }

        // step-6 : direction 2
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t r = 0; r < nq; ++r) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(q, p, k)] = th;
                    wsp0_lo[WSP(q, p, k)] = tl;
                }
            }
        }

        // step-7 : direction 1
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t q = 0; q < nq; ++q) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, j, k)] = th;
                    wsp1_lo[WSP(p, j, k)] = tl;
                }
            }
        }

        // step-8 : direction 0
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t p = 0; p < nq; ++p) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(i, j, k)] = th;
                    wsp0_lo[WSP(i, j, k)] = tl;
                }
            }
        }

        // step-9 : join wsp0 -> out
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out[DOF(i, j, k)] = ff::join(wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The same kernel with one team per element
// ---------------------------------------------------------------------------

// n rounded up to whole warps.  Every team size below goes through this:
// a team is scheduled in warps anyway, and the threads of a padded team
// beyond n simply get no iterations of the worksharing loops.  A team
// smaller than one warp is also unsafe in practice: with nvc++ 25.3 the
// plane kernel with one element per team gave nan at nq = 4 and 5 (teams
// of 16 and 25 threads) and correct results from nq = 6 (36 threads), as if
// the barriers between phases were elided inside a single warp.
constexpr int ff_round_to_warps(const int n)
{
    constexpr int warp = 32;
    return ((n + warp - 1) / warp) * warp;
}

// Threads per team of SumFactorizationFFTeams: one per point of the nq^3
// box, the largest output of any step, rounded up to whole warps.  At
// nq = 10 this is the CUDA maximum of 1024; at nq <= 4 it is one or two
// warps, which underfills an SM (several elements per team would be the
// cure).  thread_limit is an upper bound: fewer threads only mean more
// iterations per thread in the collapsed loops.
constexpr int ff_team_size(const int nq) { return ff_round_to_warps(nq * nq * nq); }

// SumFactorizationFF with the elements distributed over teams and the
// threads of a team over the points of each step: the work arrays are
// declared at team scope, so they are shared by the team (shared memory on
// a GPU) and private per element, and the end of each parallel region is
// the barrier between steps.  Every step parallelises its three output
// loops; the contraction sum stays in the thread.
template <int nq, typename index_t = int>
void SumFactorizationFFTeams(
    const std::size_t nelmt,
    const double* __restrict__ basis,
    const double* __restrict__ JxW,
    const double* __restrict__ in,
    double* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    // the team size only appears in the thread_limit clause
    [[maybe_unused]] constexpr int team_size = ff_team_size(nq);
    static_assert(team_size <= 1024, "a CUDA block holds at most 1024 threads");

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t e = 0; e < nelmt; ++e) {

        // Team-shared: the split basis and the two nq^3 boxes as hi and lo
        // arrays.  Every step assigns its full output sub-slice, so no
        // zeroing is needed.
        float B_hi[nm * nq];
        float B_lo[nm * nq];
        float wsp0_hi[nq3];
        float wsp0_lo[nq3];
        float wsp1_hi[nq3];
        float wsp1_lo[nq3];

        const double* e_in  = in  + e * nm3;
        double*       e_out = out + e * nm3;
        const double* e_JxW = JxW + e * nq3;

        #pragma omp parallel for
        for (index_t i = 0; i < nm * nq; ++i) {
            ff::split(basis[i], B_hi[i], B_lo[i]);
        }

        // step-1 : split in -> wsp0 (nm^3 sub-block of the nq^3 box)
        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    ff::split(e_in[DOF(i, j, k)], wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)]);
                }
            }
        }

        // step-2 : direction 0
        #pragma omp parallel for collapse(3)
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t i = 0; i < nm; ++i) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, j, k)] = th;
                    wsp1_lo[WSP(p, j, k)] = tl;
                }
            }
        }

        // step-3 : direction 1
        #pragma omp parallel for collapse(3)
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t j = 0; j < nm; ++j) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(q, p, k)] = th;
                    wsp0_lo[WSP(q, p, k)] = tl;
                }
            }
        }

        // step-4 : direction 2
        #pragma omp parallel for collapse(3)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t k = 0; k < nm; ++k) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, q, r)] = th;
                    wsp1_lo[WSP(p, q, r)] = tl;
                }
            }
        }

        // Reverse operations

        // step-5 : multiply with weights and determinant of Jacobi, each
        // JxW value split at its single use
        #pragma omp parallel for collapse(3)
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float jh;
                    float jl;
                    ff::split(e_JxW[WSP(p, q, r)], jh, jl);
                    ff::mul(wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)], jh, jl);
                }
            }
        }

        // step-6 : direction 2
        #pragma omp parallel for collapse(3)
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t r = 0; r < nq; ++r) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(q, p, k)] = th;
                    wsp0_lo[WSP(q, p, k)] = tl;
                }
            }
        }

        // step-7 : direction 1
        #pragma omp parallel for collapse(3)
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t q = 0; q < nq; ++q) {
                        ff::mul_acc(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
                    ff::renormalize(th, tl);
                    wsp1_hi[WSP(p, j, k)] = th;
                    wsp1_lo[WSP(p, j, k)] = tl;
                }
            }
        }

        // step-8 : direction 0
        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    float th = 0.0f;
                    float tl = 0.0f;
                    for (index_t p = 0; p < nq; ++p) {
                        ff::mul_acc(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
                    ff::renormalize(th, tl);
                    wsp0_hi[WSP(i, j, k)] = th;
                    wsp0_lo[WSP(i, j, k)] = tl;
                }
            }
        }

        // step-9 : join wsp0 -> out
        #pragma omp parallel for collapse(3)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out[DOF(i, j, k)] = ff::join(wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The plane kernels: a team of nq x nq threads per element, register columns
// ---------------------------------------------------------------------------

// Unrolling hint for the short compile-time loops over a register column:
// a column indexed by an unrolled loop counter stays in registers, one
// indexed at run time goes to local memory.
#if defined(__NVCOMPILER) || defined(__clang__)
#define FF_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define FF_UNROLL _Pragma("GCC unroll 16")
#else
#define FF_UNROLL
#endif

// Elements per team of the plane kernels.  A plane of nq x nq threads is
// one element; below a few warps per team the SM is underfilled, so low
// orders batch several elements per team.  The default fills about
// FF_PLANE_THREADS threads (14 elements at nq = 3, 8 at nq = 4, 1 from
// nq = 9); -DFF_PLANE_BATCH=n forces one batch for every order.
#ifndef FF_PLANE_THREADS
#define FF_PLANE_THREADS 128
#endif
constexpr int ff_plane_batch([[maybe_unused]] const int nq)
{
#ifdef FF_PLANE_BATCH
    return FF_PLANE_BATCH;
#else
    const int fill = FF_PLANE_THREADS / (nq * nq);
    return (fill > 1) ? fill : 1;
#endif
}

// The five phases of the plane kernels, one point of one element's plane
// each.  A thread owns the point and the whole column through it along
// the first index, held in registers as hi and lo of nq values.  Each
// phase contracts either along the register column, with no shared-memory
// traffic but the basis, read by the whole warp as a broadcast, or along
// one of the plane's directions through the team-shared boxes sm0 and sm1
// (one pair per element b of the team's batch).  The contraction index is
// the outer loop and the column index the inner one, so every input value
// and basis entry is loaded once per column rather than once per output,
// and the nq accumulation chains of a column are independent.  Every
// output is still summed in index order, so the result equals the other
// kernels' bit for bit.
//
//   A: thread (b,j,k): in(i,j,k)   -> contract i -> s[p]           -> sm0(b,p,j,k)
//   B: thread (b,q,k): sm0(b,p,j,k)-> contract j -> t[p]           -> sm1(b,p,q,k)
//   C: thread (b,q,r): sm1(b,p,q,k)-> contract k -> v[p] * JxW     -> sm0(b,p,q,r)
//   D: thread (b,q,k): sm0(b,p,q,r)-> contract r -> w[p]           -> sm1(b,p,q,k)
//   E: thread (b,j,k): sm1(b,p,q,k)-> contract q -> x[p], contract p -> out(i,j,k)
//
// Between two phases every thread of the team must have finished the
// earlier one; the kernels below provide that barrier, and the two boxes
// alternate so that no phase writes the box the previous one read.
#define SMB(b, i, j, k) ((b) * nq3 + WSP(i, j, k))

template <int nq, typename index_t>
inline void plane_phase_a(
    const index_t b, const index_t j, const index_t k,
    const double* __restrict__ e_in,
    const float* B_hi, const float* B_lo,
    float* sm0_hi, float* sm0_lo)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    float uh[nm];
    float ul[nm];
    FF_UNROLL
    for (index_t i = 0; i < nm; ++i) {
        ff::split(e_in[DOF(i, j, k)], uh[i], ul[i]);
    }
    float sh[nq];
    float sl[nq];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        sh[p] = 0.0f;
        sl[p] = 0.0f;
    }
    FF_UNROLL
    for (index_t i = 0; i < nm; ++i) {
        FF_UNROLL
        for (index_t p = 0; p < nq; ++p) {
            ff::mul_acc(sh[p], sl[p], uh[i], ul[i], B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
        }
    }
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        ff::renormalize(sh[p], sl[p]);
        sm0_hi[SMB(b, p, j, k)] = sh[p];
        sm0_lo[SMB(b, p, j, k)] = sl[p];
    }
}

template <int nq, typename index_t>
inline void plane_phase_b(
    const index_t b, const index_t q, const index_t k,
    const float* B_hi, const float* B_lo,
    const float* sm0_hi, const float* sm0_lo,
    float* sm1_hi, float* sm1_lo)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    float th[nq];
    float tl[nq];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        th[p] = 0.0f;
        tl[p] = 0.0f;
    }
    FF_UNROLL
    for (index_t j = 0; j < nm; ++j) {
        const float bh = B_hi[BAS(j, q)];
        const float bl = B_lo[BAS(j, q)];
        FF_UNROLL
        for (index_t p = 0; p < nq; ++p) {
            ff::mul_acc(th[p], tl[p], sm0_hi[SMB(b, p, j, k)], sm0_lo[SMB(b, p, j, k)], bh, bl);
        }
    }
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        ff::renormalize(th[p], tl[p]);
        sm1_hi[SMB(b, p, q, k)] = th[p];
        sm1_lo[SMB(b, p, q, k)] = tl[p];
    }
}

// Direction 2, then the weights and determinant of Jacobi, each JxW value
// split at its single use.
template <int nq, typename index_t>
inline void plane_phase_c(
    const index_t b, const index_t q, const index_t r,
    const double* __restrict__ e_JxW,
    const float* B_hi, const float* B_lo,
    const float* sm1_hi, const float* sm1_lo,
    float* sm0_hi, float* sm0_lo)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    float vh[nq];
    float vl[nq];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        vh[p] = 0.0f;
        vl[p] = 0.0f;
    }
    FF_UNROLL
    for (index_t k = 0; k < nm; ++k) {
        const float bh = B_hi[BAS(k, r)];
        const float bl = B_lo[BAS(k, r)];
        FF_UNROLL
        for (index_t p = 0; p < nq; ++p) {
            ff::mul_acc(vh[p], vl[p], sm1_hi[SMB(b, p, q, k)], sm1_lo[SMB(b, p, q, k)], bh, bl);
        }
    }
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        float jh;
        float jl;
        ff::split(e_JxW[WSP(p, q, r)], jh, jl);
        ff::renormalize(vh[p], vl[p]);
        ff::mul(vh[p], vl[p], jh, jl);
        sm0_hi[SMB(b, p, q, r)] = vh[p];
        sm0_lo[SMB(b, p, q, r)] = vl[p];
    }
}

template <int nq, typename index_t>
inline void plane_phase_d(
    const index_t b, const index_t q, const index_t k,
    const float* B_hi, const float* B_lo,
    const float* sm0_hi, const float* sm0_lo,
    float* sm1_hi, float* sm1_lo)
{
    constexpr int nq3 = nq * nq * nq;
    float wh[nq];
    float wl[nq];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        wh[p] = 0.0f;
        wl[p] = 0.0f;
    }
    FF_UNROLL
    for (index_t r = 0; r < nq; ++r) {
        const float bh = B_hi[BAS(k, r)];
        const float bl = B_lo[BAS(k, r)];
        FF_UNROLL
        for (index_t p = 0; p < nq; ++p) {
            ff::mul_acc(wh[p], wl[p], sm0_hi[SMB(b, p, q, r)], sm0_lo[SMB(b, p, q, r)], bh, bl);
        }
    }
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        ff::renormalize(wh[p], wl[p]);
        sm1_hi[SMB(b, p, q, k)] = wh[p];
        sm1_lo[SMB(b, p, q, k)] = wl[p];
    }
}

// Direction 1 back, then direction 0 back in registers, join and store the
// column out(:, j, k).
template <int nq, typename index_t>
inline void plane_phase_e(
    const index_t b, const index_t j, const index_t k,
    const float* B_hi, const float* B_lo,
    const float* sm1_hi, const float* sm1_lo,
    double* __restrict__ e_out)
{
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    float xh[nq];
    float xl[nq];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        xh[p] = 0.0f;
        xl[p] = 0.0f;
    }
    FF_UNROLL
    for (index_t q = 0; q < nq; ++q) {
        const float bh = B_hi[BAS(j, q)];
        const float bl = B_lo[BAS(j, q)];
        FF_UNROLL
        for (index_t p = 0; p < nq; ++p) {
            ff::mul_acc(xh[p], xl[p], sm1_hi[SMB(b, p, q, k)], sm1_lo[SMB(b, p, q, k)], bh, bl);
        }
    }
    float yh[nm];
    float yl[nm];
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        ff::renormalize(xh[p], xl[p]);
    }
    FF_UNROLL
    for (index_t i = 0; i < nm; ++i) {
        yh[i] = 0.0f;
        yl[i] = 0.0f;
    }
    FF_UNROLL
    for (index_t p = 0; p < nq; ++p) {
        FF_UNROLL
        for (index_t i = 0; i < nm; ++i) {
            ff::mul_acc(yh[i], yl[i], xh[p], xl[p], B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
        }
    }
    FF_UNROLL
    for (index_t i = 0; i < nm; ++i) {
        ff::renormalize(yh[i], yl[i]);
        e_out[DOF(i, j, k)] = ff::join(yh[i], yl[i]);
    }
}

#undef SMB

// The plane kernel with teams distribute: batch elements per team of
// batch * nq * nq threads, rounded up to whole warps (ff_round_to_warps),
// one parallel region per team and a worksharing loop per phase, whose
// implicit barrier separates the phases.
template <int nq, int batch = 1, typename index_t = int>
void SumFactorizationFFPlane(
    const std::size_t nelmt,
    const double* __restrict__ basis,
    const double* __restrict__ JxW,
    const double* __restrict__ in,
    double* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    // the team size only appears in the thread_limit clause
    [[maybe_unused]] constexpr int team_size = ff_round_to_warps(batch * nq * nq);
    static_assert(team_size <= 1024, "a CUDA block holds at most 1024 threads");

    const std::size_t nteams = (nelmt + batch - 1) / batch;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams distribute thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the split basis and two nq^3 boxes per element.
        float B_hi[nm * nq];
        float B_lo[nm * nq];
        float sm0_hi[batch * nq3];
        float sm0_lo[batch * nq3];
        float sm1_hi[batch * nq3];
        float sm1_lo[batch * nq3];

        // The team's elements e0 .. e0 + batch - 1; the phase loops run over
        // the full batch with a constant trip count, so that the compiler
        // can collapse the batch into the thread index, and the guard skips
        // the missing elements of a short last team.
        const std::size_t e0 = t * batch;

        #pragma omp parallel
        {
            #pragma omp for
            for (index_t i = 0; i < nm * nq; ++i) {
                ff::split(basis[i], B_hi[i], B_lo[i]);
            }

            #pragma omp for collapse(3)
            for (index_t b = 0; b < batch; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        if (e0 + b < nelmt) {
                            plane_phase_a<nq>(b, j, k, in + (e0 + b) * nm3, B_hi, B_lo, sm0_hi, sm0_lo);
                        }
                    }
                }
            }

            #pragma omp for collapse(3)
            for (index_t b = 0; b < batch; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        if (e0 + b < nelmt) {
                            plane_phase_b<nq>(b, q, k, B_hi, B_lo, sm0_hi, sm0_lo, sm1_hi, sm1_lo);
                        }
                    }
                }
            }

            #pragma omp for collapse(3)
            for (index_t b = 0; b < batch; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t r = 0; r < nq; ++r) {
                        if (e0 + b < nelmt) {
                            plane_phase_c<nq>(b, q, r, JxW + (e0 + b) * nq3, B_hi, B_lo,
                                              sm1_hi, sm1_lo, sm0_hi, sm0_lo);
                        }
                    }
                }
            }

            #pragma omp for collapse(3)
            for (index_t b = 0; b < batch; ++b) {
                for (index_t q = 0; q < nq; ++q) {
                    for (index_t k = 0; k < nm; ++k) {
                        if (e0 + b < nelmt) {
                            plane_phase_d<nq>(b, q, k, B_hi, B_lo, sm0_hi, sm0_lo, sm1_hi, sm1_lo);
                        }
                    }
                }
            }

            #pragma omp for collapse(3) nowait
            for (index_t b = 0; b < batch; ++b) {
                for (index_t j = 0; j < nm; ++j) {
                    for (index_t k = 0; k < nm; ++k) {
                        if (e0 + b < nelmt) {
                            plane_phase_e<nq>(b, j, k, B_hi, B_lo, sm1_hi, sm1_lo, out + (e0 + b) * nm3);
                        }
                    }
                }
            }
        }
    }
}

// The same plane kernel with the loop construct: teams loop over the
// teams and a nested loop per phase, no parallel region and no
// worksharing loop.  This is the descriptive form nvc++ lowers like
// OpenACC gang and vector loops: one kernel function, the team-scope
// arrays in shared memory, and the barrier between two nested loops
// placed by the compiler.  Compilers that bind a nested loop to the
// team's initial thread (g++, clang without an enclosing parallel region)
// run the phases of a team on one thread, which is correct and slow.
//
// nvc++ 25.3 spreads a phase loop over the threads only when its constant
// trip count exceeds 16: with one element per team at nq <= 4 the phases
// have 9 to 16 iterations and the whole element runs on one thread of the
// team (1.5 GDoF/s at nq = 4 on Grace Hopper against 12 with a runtime
// bound).  The default batch of ff_plane_batch never meets this; a batch
// of 2 or more at nq = 4 does not either.
template <int nq, int batch = 1, typename index_t = int>
void SumFactorizationFFPlaneLoop(
    const std::size_t nelmt,
    const double* __restrict__ basis,
    const double* __restrict__ JxW,
    const double* __restrict__ in,
    double* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    static_assert(batch >= 1, "at least one element per team");
    constexpr int nm = nq - 1;
    constexpr int nq3 = nq * nq * nq;
    constexpr int nm3 = nm * nm * nm;
    // the team size only appears in the thread_limit clause
    [[maybe_unused]] constexpr int team_size = ff_round_to_warps(batch * nq * nq);
    static_assert(team_size <= 1024, "a CUDA block holds at most 1024 threads");

    const std::size_t nteams = (nelmt + batch - 1) / batch;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq3], in[:nelmt*nm3]) \
        map(from: out[:nelmt*nm3])
    #pragma omp teams loop thread_limit(team_size)
    for (std::size_t t = 0; t < nteams; ++t) {

        // Team-shared: the split basis and two nq^3 boxes per element.
        // nvc++ 25.3 places these in CUDA shared memory only up to about
        // 8 KB in total (nq <= 7 with one element per team) and elsewhere
        // in global memory behind L1; it does not support the allocate
        // directive that would request shared memory.  On Grace Hopper the
        // placement made no measurable difference at nq = 5..7.
        float B_hi[nm * nq];
        float B_lo[nm * nq];
        float sm0_hi[batch * nq3];
        float sm0_lo[batch * nq3];
        float sm1_hi[batch * nq3];
        float sm1_lo[batch * nq3];

        // The team's elements e0 .. e0 + batch - 1; the phase loops run over
        // the full batch with a constant trip count, so that the compiler
        // can collapse the batch into the thread index, and the guard skips
        // the missing elements of a short last team.
        const std::size_t e0 = t * batch;

        #pragma omp loop
        for (index_t i = 0; i < nm * nq; ++i) {
            ff::split(basis[i], B_hi[i], B_lo[i]);
        }

        #pragma omp loop collapse(3)
        for (index_t b = 0; b < batch; ++b) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    if (e0 + b < nelmt) {
                        plane_phase_a<nq>(b, j, k, in + (e0 + b) * nm3, B_hi, B_lo, sm0_hi, sm0_lo);
                    }
                }
            }
        }

        #pragma omp loop collapse(3)
        for (index_t b = 0; b < batch; ++b) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t k = 0; k < nm; ++k) {
                    if (e0 + b < nelmt) {
                        plane_phase_b<nq>(b, q, k, B_hi, B_lo, sm0_hi, sm0_lo, sm1_hi, sm1_lo);
                    }
                }
            }
        }

        #pragma omp loop collapse(3)
        for (index_t b = 0; b < batch; ++b) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {
                    if (e0 + b < nelmt) {
                        plane_phase_c<nq>(b, q, r, JxW + (e0 + b) * nq3, B_hi, B_lo,
                                          sm1_hi, sm1_lo, sm0_hi, sm0_lo);
                    }
                }
            }
        }

        #pragma omp loop collapse(3)
        for (index_t b = 0; b < batch; ++b) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t k = 0; k < nm; ++k) {
                    if (e0 + b < nelmt) {
                        plane_phase_d<nq>(b, q, k, B_hi, B_lo, sm0_hi, sm0_lo, sm1_hi, sm1_lo);
                    }
                }
            }
        }

        #pragma omp loop collapse(3)
        for (index_t b = 0; b < batch; ++b) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    if (e0 + b < nelmt) {
                        plane_phase_e<nq>(b, j, k, B_hi, B_lo, sm1_hi, sm1_lo, out + (e0 + b) * nm3);
                    }
                }
            }
        }
    }
}

#undef FF_UNROLL

#undef WSP
#undef DOF
#undef BAS

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

// The double test data as seen by precision T: a reference for T = double,
// otherwise a rounded (float) copy held in `storage`.
template <typename T>
const std::vector<T>& as_precision(const std::vector<double>& v, std::vector<T>& storage)
{
    if constexpr (std::is_same_v<T, double>) {
        return v;
    } else {
        storage.assign(v.begin(), v.end());
        return storage;
    }
}

// ||x - ref|| / ||ref||, every x[i] widened to double, both sums accumulated
// in double.
template <typename T>
double relative_error(const std::vector<T>& x, const std::vector<double>& ref)
{
    assert(x.size() == ref.size());
    double num = 0;
    double den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(x[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

// The rates of one timed kernel, for the CSV row.
struct Rates {
    double gdofs;
    double gbs;
    double norm;
};

// Time `kernel` on the data in precision T, print its row and return its
// output; the rates go to `rates`.
template <typename T, int nq>
std::vector<T> run_precision(
    const char* label, const sumfact_kernel<T> kernel,
    const std::size_t nelmt, const int ntests,
    const std::vector<double>& basis64,
    const std::vector<double>& JxW64,
    const std::vector<double>& in64,
    Rates& rates)
{
    std::vector<T> basis_T;
    std::vector<T> JxW_T;
    std::vector<T> in_T;
    const std::vector<T>& basis = as_precision<T>(basis64, basis_T);
    const std::vector<T>& JxW   = as_precision<T>(JxW64, JxW_T);
    const std::vector<T>& in    = as_precision<T>(in64, in_T);
    std::vector<T> out(in.size());

    const double elapsed = time_sumfact<T, nq>(
        nelmt, ntests, basis.data(), JxW.data(), in.data(), out.data(), kernel);

    // GDoF/s, and GB/s for read in + write out + read JxW at sizeof(T) bytes
    const double dof_rate  = 1.0e-9 * in.size() / elapsed;
    const double byte_rate = 1.0e-9 * sizeof(T) * (2 * in.size() + JxW.size()) / elapsed;

    std::cout << std::left << std::setw(12) << label << std::right
              << " GDoF/s = " << std::setw(9) << dof_rate
              << " GB/s = "   << std::setw(9) << byte_rate
              << " norm = "   << std::setprecision(10) << norm2(out) << std::setprecision(6)
              << "\n";
    rates = Rates{dof_rate, byte_rate, norm2(out)};
    return out;
}

// One float-float kernel of run_test: its row, its error against the double
// result and its CSV row (BK_CSV).
template <int nq>
double run_kernel(const char* label, const sumfact_kernel<double> kernel, const int batch,
                  const std::size_t nelmt, const int ntests, const double tol,
                  const std::vector<double>& basis, const std::vector<double>& JxW,
                  const std::vector<double>& in, const std::vector<double>& ref)
{
    Rates rates{};
    const double err = relative_error(
        run_precision<double, nq>(label, kernel, nelmt, ntests, basis, JxW, in, rates), ref);
    append_csv_row("omp", label, nq, nelmt, batch, rates.gdofs, rates.gbs, rates.norm, err, err <= tol);
    return err;
}

// Returns true when the float-float result is within tol of the double one.
template <int nq>
bool run_test(const std::size_t nelmt, const int ntests, const bool random_data,
              const double tol)
{
    constexpr int nm = nq - 1;

    const std::array<double, nm * nq> b = make_test_basis<double, nm, nq>();
    const std::vector<double> basis(b.begin(), b.end());
    std::vector<double> JxW(nelmt * nq * nq * nq, 1.0);
    std::vector<double> in (nelmt * nm * nm * nm, 3.0);
    if (random_data) {
        fill_random_test_data(in, JxW);
    }

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " nq = " << nq
              << " data = " << (random_data ? "random" : "constant")
              << " fma = " << (ff::hardware_fma ? "hardware" : "library call")
              << "\n";

    Rates rates{};
    const std::vector<double> ref = run_precision<double, nq>(
        "double", &SumFactorization<double, nq>, nelmt, ntests, basis, JxW, in, rates);
    append_csv_row("omp", "double", nq, nelmt, 0, rates.gdofs, rates.gbs, rates.norm, std::nullopt, true);
    const double err_float = relative_error(run_precision<float, nq>(
        "float", &SumFactorization<float, nq>, nelmt, ntests, basis, JxW, in, rates), ref);
    append_csv_row("omp", "float", nq, nelmt, 0, rates.gdofs, rates.gbs, rates.norm, err_float, true);
    constexpr int batch = ff_plane_batch(nq);
    const std::string batch_label = "ff-plane x" + std::to_string(batch);
    const std::string loop_label = "ff-loop x" + std::to_string(batch);
    const double err_ff = run_kernel<nq>(
        "float-float", &SumFactorizationFF<nq>, 0, nelmt, ntests, tol, basis, JxW, in, ref);
    const double err_teams = run_kernel<nq>(
        "ff-teams", &SumFactorizationFFTeams<nq>, 1, nelmt, ntests, tol, basis, JxW, in, ref);
    const double err_plane = run_kernel<nq>(
        "ff-plane", &SumFactorizationFFPlane<nq, 1>, 1, nelmt, ntests, tol, basis, JxW, in, ref);
    const double err_batch = run_kernel<nq>(
        batch_label.c_str(), &SumFactorizationFFPlane<nq, batch>, batch, nelmt, ntests, tol, basis, JxW, in, ref);
    const double err_loop = run_kernel<nq>(
        "ff-loop", &SumFactorizationFFPlaneLoop<nq, 1>, 1, nelmt, ntests, tol, basis, JxW, in, ref);
    const double err_loop_batch = run_kernel<nq>(
        loop_label.c_str(), &SumFactorizationFFPlaneLoop<nq, batch>, batch, nelmt, ntests, tol, basis, JxW, in, ref);

    const bool ok = (err_ff <= tol) && (err_teams <= tol) && (err_plane <= tol) && (err_batch <= tol)
                 && (err_loop <= tol) && (err_loop_batch <= tol);
    std::cout << "float        rel. error vs double = " << err_float << "\n"
              << (ok ? "ok" : "FAIL")
              << ": float-float rel. error vs double = " << err_ff
              << ", ff-teams = " << err_teams
              << ", ff-plane = " << err_plane
              << ", x" << batch << " = " << err_batch
              << ", ff-loop = " << err_loop
              << ", x" << batch << " = " << err_loop_batch
              << " (tol = " << tol << ")\n";
    return ok;
}

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288, as BK1

} // namespace

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const bool random_data = get_env("BK_RANDOM").has_value();
    const double tol = std::atof(get_env("BK_TOL").value_or("1e-10").c_str());

    // Runtime p -> compile-time nq = p + 2, one instantiation per order.
    bool ok = false;
    switch (p) {
        case 1: ok = run_test< 3>(nelmt, ntests, random_data, tol); break;
        case 2: ok = run_test< 4>(nelmt, ntests, random_data, tol); break;
        case 3: ok = run_test< 5>(nelmt, ntests, random_data, tol); break;
        case 4: ok = run_test< 6>(nelmt, ntests, random_data, tol); break;
        case 5: ok = run_test< 7>(nelmt, ntests, random_data, tol); break;
        case 6: ok = run_test< 8>(nelmt, ntests, random_data, tol); break;
        case 7: ok = run_test< 9>(nelmt, ntests, random_data, tol); break;
        case 8: ok = run_test<10>(nelmt, ntests, random_data, tol); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return ok ? 0 : 1;
}
