// BK1 (mass matrix) with OpenACC offloading: one driver, several kernels.
// BK_KERNEL=<name> selects the kernel at run time (default: elem):
//
//   elem      `parallel loop` over elements, the compiler's schedule (gang and
//             vector over elements); the element's two work boxes are private
//             to its thread, the phases are plain loops inside it.
//   gang1..3  `parallel loop gang` over elements, each phase nest under
//             `loop vector collapse(C)`; the boxes are gang-private (shared
//             memory). C = 1, 2, 3 collapsed levels; the lanes that are left
//             over at C < 3 run the remaining levels sequentially.
//   worker    gang over batches of W elements, `loop worker` over the batch's
//             elements, `loop vector collapse(3)` per phase: several elements
//             per block without touching the phase loops.
//   batch     gang over batches of EPG elements, every phase one flat vector
//             loop over the fibres of the batch, one lane per fibre with the
//             fibre held in registers (the dealii-X/benchmarks structure).
//
// The per-element kernels (elem, gang, worker) share one phase body
// (BK1_PHASES) over two nq^3 boxes w0 and w1 with the fastest index last.
// Each phase's innermost loop is the fastest index of the array it writes,
// so when the nest is collapsed onto vector lanes the lane-adjacent accesses
// are stride-1; the contraction loop is sequential inside. JxW is read as
// JxW(p,q,r) at p*nq^2 + q*nq + r, as in BK1.cpp. The batch kernel keeps its
// own phase body and layouts (table before the kernel), and reads JxW in the
// order it writes, r*nq^2 + p*nq + q; the test data does not depend on the
// convention, but random-data norms of batch and the others differ.
//
// Environment: BK_KERNEL selects the kernel; BK_RANDOM=1 fills in and JxW
// with seeded random data (see acc_common.h); the default is the constant
// data of BK1.cpp. Build-time knobs (make -C acc DEFS=-D...): BK_CACHE_BASIS
// for every kernel, and BK_TARGET_LANES, BK_MAX_SCRATCH_KB, BK_EPG for the
// batch kernel (see batch_elements). The driver prints the kernel's
// configuration on stderr.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acc_common.h"

namespace bk {

// ---------------------------------------------------------------------------
// Pragma helpers. nvc++ accepts a template parameter as the collapse
// argument; GCC's OpenACC front end wants a literal, so its host fallback,
// which runs every loop serially anyway, gets the clause dropped.
// ---------------------------------------------------------------------------
#define BK_PRAGMA(x) _Pragma(#x)
#ifdef __NVCOMPILER
#define BK_LOOP_VECTOR_COLLAPSE(C) BK_PRAGMA(acc loop vector collapse(C))
#else
#define BK_LOOP_VECTOR_COLLAPSE(C) BK_PRAGMA(acc loop vector)
#endif

// BK_CACHE_BASIS (make -C acc DEFS=-DBK_CACHE_BASIS) stamps `cache(readonly:
// basis)` at the top of the element or batch loop of every kernel but
// worker (see there): the basis is fetched into shared memory for the loop
// body instead of being left to L1, and the readonly modifier (OpenACC 2.7)
// tells the compiler no write-back is needed. Measured on GH200 it changes
// nothing, within a few percent either way, at any order of elem, gang3 or
// batch.
#if defined(BK_CACHE_BASIS) && defined(__NVCOMPILER)
#define BK1_CACHE_BASIS BK_PRAGMA(acc cache(readonly: basis[0:nm*nq]))
#elif defined(BK_CACHE_BASIS)
#define BK1_CACHE_BASIS BK_PRAGMA(acc cache(basis[0:nm*nq]))   /* GCC: no readonly modifier */
#else
#define BK1_CACHE_BASIS
#endif

constexpr int round_to_warps(const int lanes)
{
    constexpr int warp = 32;
    constexpr int max_lanes = 1024;
    const int rounded = ((lanes + warp - 1) / warp) * warp;
    return rounded > max_lanes ? max_lanes : rounded;
}

// ---------------------------------------------------------------------------
// The phases of one element. LOOP is the pragma stamped before each phase
// nest: `loop vector collapse(C)` or nothing (elem). The enclosing scope provides
// T, nq, nm, index_t, the pointers of BK1_ELEMENT_POINTERS and `live`
// (whether the store happens); each lane body opens with BK1_ACCESSORS, the
// lambdas e_in(i,j,k), e_out(i,j,k), e_JxW(p,q,r), B(i,p), w0(a,b,c), w1(a,b,c).
// Step 1 stays a copy: step 2 reads every input nq times, once per output,
// and it should read the box, not global memory. Step 9 is fused into
// step 8, whose outputs are each stored once.
//
//   step  reads          contracts  writes          loops (innermost last)
//   1     in  (i,j,k)    -          w0 (i,j,k)      i, j, k
//   2     w0  (i,j,k)    i -> p     w1 (p,j,k)      p, j, k
//   3     w1  (p,j,k)    j -> q     w0 (q,p,k)      q, p, k
//   4+5   w0  (q,p,k)    k -> r     w1 (p,q,r)*JxW  p, q, r
//   6     w1  (p,q,r)    r -> k     w0 (q,p,k)      q, p, k
//   7     w0  (q,p,k)    q -> j     w1 (p,j,k)      p, j, k
//   8+9   w1  (p,j,k)    p -> i     out (i,j,k)     i, j, k
// ---------------------------------------------------------------------------
#define BK1_PHASES(LOOP)                                                     \
    /* step-1 : copy in -> w0 (nm^3 sub-block of the nq^3 box) */            \
    LOOP                                                                     \
    for (index_t i = 0; i < nm; ++i) {                                       \
        for (index_t j = 0; j < nm; ++j) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                w0(i, j, k) = e_in(i, j, k);                                 \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-2 : direction 0 */                                               \
    LOOP                                                                     \
    for (index_t p = 0; p < nq; ++p) {                                       \
        for (index_t j = 0; j < nm; ++j) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t i = 0; i < nm; ++i) {                           \
                    tmp += w0(i, j, k) * B(i, p);                            \
                }                                                            \
                w1(p, j, k) = tmp;                                           \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-3 : direction 1 */                                               \
    LOOP                                                                     \
    for (index_t q = 0; q < nq; ++q) {                                       \
        for (index_t p = 0; p < nq; ++p) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t j = 0; j < nm; ++j) {                           \
                    tmp += w1(p, j, k) * B(j, q);                            \
                }                                                            \
                w0(q, p, k) = tmp;                                           \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-4 : direction 2, and step-5 : weights and Jacobian */            \
    LOOP                                                                     \
    for (index_t p = 0; p < nq; ++p) {                                       \
        for (index_t q = 0; q < nq; ++q) {                                   \
            for (index_t r = 0; r < nq; ++r) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t k = 0; k < nm; ++k) {                           \
                    tmp += w0(q, p, k) * B(k, r);                            \
                }                                                            \
                w1(p, q, r) = tmp * e_JxW(p, q, r);                          \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-6 : direction 2 */                                               \
    LOOP                                                                     \
    for (index_t q = 0; q < nq; ++q) {                                       \
        for (index_t p = 0; p < nq; ++p) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t r = 0; r < nq; ++r) {                           \
                    tmp += w1(p, q, r) * B(k, r);                            \
                }                                                            \
                w0(q, p, k) = tmp;                                           \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-7 : direction 1 */                                               \
    LOOP                                                                     \
    for (index_t p = 0; p < nq; ++p) {                                       \
        for (index_t j = 0; j < nm; ++j) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t q = 0; q < nq; ++q) {                           \
                    tmp += w0(q, p, k) * B(j, q);                            \
                }                                                            \
                w1(p, j, k) = tmp;                                           \
            }                                                                \
        }                                                                    \
    }                                                                        \
    /* step-8 : direction 0, and step-9 : the store to out. `live` is false  \
       only for a worker past the end of the element range */                \
    LOOP                                                                     \
    for (index_t i = 0; i < nm; ++i) {                                       \
        for (index_t j = 0; j < nm; ++j) {                                   \
            for (index_t k = 0; k < nm; ++k) {                               \
                BK1_ACCESSORS                                                \
                T tmp = 0;                                                   \
                for (index_t p = 0; p < nq; ++p) {                           \
                    tmp += w1(p, j, k) * B(i, p);                            \
                }                                                            \
                if (live) {                                                  \
                    e_out(i, j, k) = tmp;                                    \
                }                                                            \
            }                                                                \
        }                                                                    \
    }

// Accessors of one element for the per-element kernels, in two parts.
// BK1_ELEMENT_POINTERS(e) sets the element's base pointers: the blocks of
// in, out and JxW (BK1.cpp's layouts) and the two nq^3 work boxes w0s and
// w1s. BK1_ACCESSORS declares the lambdas over them, each holding its
// pointer by value, and is stamped at the top of every phase's lane body:
// declared at gang level the closures are gang-private objects that nvc++
// places in shared memory next to the boxes, so every access reloads the
// pointer from shared memory (the stores through the other box may alias
// it); declared per lane they are registers.
#define BK1_ELEMENT_POINTERS(e)                                              \
    T* const       w0p   = w0s;                                              \
    T* const       w1p   = w1s;                                              \
    const T* const in_e  = in  + (e) * (nm * nm * nm);                       \
    T* const       out_e = out + (e) * (nm * nm * nm);                       \
    const T* const JxW_e = JxW + (e) * (nq * nq * nq);

#define BK1_ACCESSORS                                                        \
    [[maybe_unused]] const auto B = [basis](index_t i, index_t p) -> const T& {               \
        return basis[i * nq + p];                                            \
    };                                                                       \
    [[maybe_unused]] const auto e_in = [in_e](index_t i, index_t j, index_t k) -> const T& {  \
        return in_e[i * (nm * nm) + j * nm + k];                             \
    };                                                                       \
    [[maybe_unused]] const auto e_out = [out_e](index_t i, index_t j, index_t k) -> T& {      \
        return out_e[i * (nm * nm) + j * nm + k];                            \
    };                                                                       \
    [[maybe_unused]] const auto e_JxW = [JxW_e](index_t p, index_t q, index_t r) -> const T& { \
        return JxW_e[p * (nq * nq) + q * nq + r];                            \
    };                                                                       \
    [[maybe_unused]] const auto w0 = [w0p](index_t a, index_t b, index_t c) -> T& {           \
        return w0p[a * (nq * nq) + b * nq + c];                              \
    };                                                                       \
    [[maybe_unused]] const auto w1 = [w1p](index_t a, index_t b, index_t c) -> T& {           \
        return w1p[a * (nq * nq) + b * nq + c];                              \
    };

// The data clauses of every kernel: no-ops when the arrays are already
// present (the driver's `acc data` region), and they make the kernels usable
// on their own.
#define BK1_DATA_CLAUSES                                                     \
    copyin(basis[0:nm*nq])                                                   \
    copyin(JxW[0:nelmt*nq*nq*nq], in[0:nelmt*nm*nm*nm])                      \
    copyout(out[0:nelmt*nm*nm*nm])

// ---------------------------------------------------------------------------
// elem: one thread per element, the compiler's schedule for the element loop
// ---------------------------------------------------------------------------
template <typename T, int nq, typename index_t = int>
void SumFactorizationElem(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;

    #pragma acc parallel loop BK1_DATA_CLAUSES
    for (std::size_t e = 0; e < nelmt; ++e) {
        BK1_CACHE_BASIS
        constexpr bool live = true;
        T w0s[nq * nq * nq];
        T w1s[nq * nq * nq];
        BK1_ELEMENT_POINTERS(e)

        // No directives inside the element loop: the phases are plain loops
        // with constant trip counts that the compiler unrolls, and at low
        // order it keeps the boxes in registers. An explicit `loop seq` on
        // them is not harmless: it is what made this kernel slow at p = 1.
        BK1_PHASES()
    }
}

// ---------------------------------------------------------------------------
// gang<C>: one gang per element, each phase collapsed C levels onto the
// vector lanes. The vector length is the largest phase trip count at that C,
// nq^C, rounded to warps.
// ---------------------------------------------------------------------------
constexpr int gang_vector_length(const int nq, const int C)
{
    int lanes = 1;
    for (int c = 0; c < C; ++c) {
        lanes *= nq;
    }
    return round_to_warps(lanes);
}

template <typename T, int nq, int C = 3, typename index_t = int>
void SumFactorizationGang(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(C >= 1 && C <= 3, "collapse 1, 2 or 3 levels of a phase nest");
    constexpr int nm = nq - 1;
    [[maybe_unused]] constexpr int vlen = gang_vector_length(nq, C);   // only the pragma uses it

    #pragma acc parallel loop gang vector_length(vlen) BK1_DATA_CLAUSES
    for (std::size_t e = 0; e < nelmt; ++e) {
        BK1_CACHE_BASIS
        constexpr bool live = true;
        T w0s[nq * nq * nq];
        T w1s[nq * nq * nq];
        BK1_ELEMENT_POINTERS(e)

        BK1_PHASES(BK_LOOP_VECTOR_COLLAPSE(C))
    }
}

// ---------------------------------------------------------------------------
// worker: W elements per gang, one worker each, the phases collapsed onto
// the worker's vector lanes. The vector length is capped at 512 so that
// every order gets at least two workers; W * V <= 1024.
// ---------------------------------------------------------------------------
constexpr int worker_vector_length(const int nq)
{
    const int lanes = gang_vector_length(nq, 3);
    return lanes > 512 ? 512 : lanes;
}
constexpr int worker_count(const int nq) { return 1024 / worker_vector_length(nq); }

template <typename T, int nq, typename index_t = int>
void SumFactorizationWorker(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;
    constexpr int W = worker_count(nq);
    [[maybe_unused]] constexpr int V = worker_vector_length(nq);   // only the pragma uses it
    const std::size_t ngang = (nelmt + W - 1) / W;

    // No BK1_CACHE_BASIS here: a cache directive above the worker loop is
    // legal OpenACC, but nvc++ 25.3 (NVHPC, linuxarm64) miscompiles it
    // (illegal instruction at nq = 4, misaligned address at nq = 7..10, on
    // GH200), while the same phase body runs correctly with the directive
    // in the other kernels and without it here.
    #pragma acc parallel loop gang num_workers(W) vector_length(V) BK1_DATA_CLAUSES
    for (std::size_t g = 0; g < ngang; ++g) {
        #pragma acc loop worker
        for (int w = 0; w < W; ++w) {
            // A worker past the end recomputes the last element and skips the
            // store: every worker must run every phase, since the barriers
            // between the phases are block-wide.
            const std::size_t e0   = g * W + w;
            const bool        live = e0 < nelmt;
            const std::size_t e    = live ? e0 : nelmt - 1;

            // Worker-private boxes. nvc++ 25.3 reports no shared-memory
            // placement for them (nor for a gang-level [W][nq^3] array with a
            // slice per worker, which measured 2.5 to 6 times slower still),
            // so unlike the gang kernels' boxes they live outside shared
            // memory and this kernel runs below gang3 at most orders.
            T w0s[nq * nq * nq];
            T w1s[nq * nq * nq];
            BK1_ELEMENT_POINTERS(e)

            BK1_PHASES(BK_LOOP_VECTOR_COLLAPSE(3))
        }
    }
}

// ---------------------------------------------------------------------------
// batch: a batch of EPG elements per gang, one lane per fibre.
//
// Every phase is one vector loop over the fibres of the whole batch; a lane
// loads the inputs along the contraction direction into registers (reg[])
// and forms every output of that line, so each shared-memory value is read
// once per phase instead of once per output. The explicit copy matters:
// with the loads written in place of reg[], nvc++ reloads them for every
// output and the kernel measured up to 2.3x slower (nq = 8, 1e6 elements).
//
// The two work arrays are boxes of nq^3 entries per element in every phase;
// a phase that produces nm values along an index fills the first nm. Every
// phase writes with the lane index as the fastest box index. Per phase,
// fastest index last:
//
//   step  reads              contracts  writes              lanes over
//   1+2   in   (e,i,j,k)     i -> p     wsp1 (e,p,j,k)      (e,j,k)
//   3     wsp1 (e,p,j,k)     j -> q     wsp0 (e,p,q,k)      (e,p,k)
//   4+5   wsp0 (e,p,q,k)     k -> r     wsp1 (e,r,p,q)*JxW  (e,p,q)  loads stride nq across lanes
//   6     wsp1 (e,r,p,q)     r -> k     wsp0 (e,k,p,q)      (e,p,q)
//   7     wsp0 (e,k,p,q)     q -> j     wsp1 (e,j,p,k)      (e,p,k)  loads stride nq across lanes
//   8+9   wsp1 (e,j,p,k)     p -> i     out  (e,i,j,k)      (e,j,k)
//
// Measured on GH200 with 1e6 elements: at nq = 9 and 10 the best batch is 3
// elements (256 and 320 lanes), and 512 lanes is the worst of 2..6 elements
// at both orders. Padding the fastest box extent to an odd stride, to clear
// the bank conflicts of the strided loads in steps 4 and 7, lost 5 to 21
// percent at nq = 4, 8 and 10: it breaks the stride-1 runs of every other
// access into 2-way conflicts, and those are the majority.
//
// Batch configuration, overridable at build time (make -C acc DEFS=-D...):
//   BK_TARGET_LANES   fibres per gang the batch size aims for (default 512)
//   BK_MAX_SCRATCH_KB cap on the two work boxes, the static shared-memory
//                     allocation of a block is 48 KB (default 48)
//   BK_EPG            fixed elements per gang for every order, overriding
//                     the two above
// ---------------------------------------------------------------------------
#ifndef BK_TARGET_LANES
#define BK_TARGET_LANES 512
#endif
#ifndef BK_MAX_SCRATCH_KB
#define BK_MAX_SCRATCH_KB 48
#endif

// Elements per gang: enough fibres (nq^2 per element) to reach the lane
// target, capped by the shared-memory budget. With the defaults the cap
// never binds for nq <= 10: nq = 4: 32 elements (16 KB); nq = 8: 8 (32 KB);
// nq = 9: 6 (34 KB); nq = 10: 5 (40 KB).
constexpr int batch_elements(const int nq)
{
#ifdef BK_EPG
    return BK_EPG;
#else
    const int by_lanes = BK_TARGET_LANES / (nq * nq);
    const int by_shared = (BK_MAX_SCRATCH_KB * 1024) / (2 * nq * nq * nq * int(sizeof(float)));
    const int n = by_lanes < by_shared ? by_lanes : by_shared;
    return n < 1 ? 1 : n;
#endif
}

// Vector length: the batch's fibre count rounded up to whole warps.
constexpr int batch_vector_length(const int nq, const int epg)
{
    return round_to_warps(epg * nq * nq);
}

// Accessors of the batch kernel, stamped at the top of every phase's lane
// body (see BK1_ACCESSORS for why not at gang level). Each returns the entry
// by reference through its linear index, index times stride with compile-
// time strides, and holds its base pointer by value. w0/w1 are the nq^3
// boxes of the work arrays; in_e/out_e the nm^3 blocks of the batch's
// elements; W the weights of the batch; B the basis, B(i, p) == basis[i*nq+p].
#define BK1_BATCH_ACCESSORS                                                  \
    [[maybe_unused]] const auto w0 = [w0p](index_t e, index_t a, index_t b, index_t c) -> T& { \
        return w0p[e * box + a * (nq * nq) + b * nq + c];                    \
    };                                                                       \
    [[maybe_unused]] const auto w1 = [w1p](index_t e, index_t a, index_t b, index_t c) -> T& { \
        return w1p[e * box + a * (nq * nq) + b * nq + c];                    \
    };                                                                       \
    [[maybe_unused]] const auto in_e = [in_b](index_t e, index_t i, index_t j, index_t k) -> const T& { \
        return in_b[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];        \
    };                                                                       \
    [[maybe_unused]] const auto out_e = [out_b](index_t e, index_t i, index_t j, index_t k) -> T& { \
        return out_b[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];       \
    };                                                                       \
    [[maybe_unused]] const auto W = [JxW_b](index_t e, index_t r, index_t p, index_t q) -> const T& { \
        return JxW_b[e * box + r * (nq * nq) + p * nq + q];                  \
    };                                                                       \
    [[maybe_unused]] const auto B = [basis](index_t i, index_t p) -> const T& {               \
        return basis[i * nq + p];                                            \
    };

template <typename T, int nq, int EPG = batch_elements(nq), typename index_t = int>
void SumFactorizationBatch(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(EPG >= 1, "at least one element per gang");
    constexpr int nm = nq - 1;
    [[maybe_unused]] constexpr int vlen = batch_vector_length(nq, EPG);   // only the pragma uses it
    constexpr int box = nq * nq * nq;             // work array entries per element
    static_assert(2 * EPG * box * int(sizeof(T)) <= 48 * 1024,
                  "the two work boxes must fit the static shared memory of a block");

    const std::size_t ngang = (nelmt + EPG - 1) / EPG;

    #pragma acc parallel loop gang vector_length(vlen) BK1_DATA_CLAUSES
    for (std::size_t g = 0; g < ngang; ++g) {
        BK1_CACHE_BASIS

        // Gang-private work arrays (shared memory on the GPU). The phases
        // alternate between them: a phase writes a layout transposed from
        // the one it reads, so it cannot run in place.
        T wsp0[EPG * box];
        T wsp1[EPG * box];

        const std::size_t e0 = g * EPG;
        // Elements in this batch; the last batch may be short. The phases
        // shrink their trip counts, the barriers between them are outside
        // the loops, so every lane still reaches every barrier.
        const index_t nb = (nelmt - e0 < std::size_t(EPG)) ? index_t(nelmt - e0) : EPG;

        // Base pointers of the batch (gang-private scalars, registers): the
        // boxes, the batch's blocks of in and out, and its weights. The
        // accessor lambdas over them, BK1_BATCH_ACCESSORS, are declared at
        // the top of every phase's lane body for the reason given at
        // BK1_ACCESSORS: a gang-level closure sits in shared memory and its
        // pointer is reloaded on every access.
        T* const       w0p   = wsp0;
        T* const       w1p   = wsp1;
        const T* const in_b  = in  + e0 * (nm * nm * nm);
        T* const       out_b = out + e0 * (nm * nm * nm);
        const T* const JxW_b = JxW + e0 * box;

        // step-1 : read the input fibre, fused with
        // step-2 : direction 0, one lane per fibre (e, j, k)
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    BK1_BATCH_ACCESSORS
                    T reg[nm];
                    for (index_t i = 0; i < nm; ++i) {
                        reg[i] = in_e(e, i, j, k);
                    }
                    for (index_t p = 0; p < nq; ++p) {
                        T tmp = 0;
                        for (index_t i = 0; i < nm; ++i) {
                            tmp += reg[i] * B(i, p);
                        }
                        w1(e, p, j, k) = tmp;
                    }
                }
            }
        }

        // step-3 : direction 1, one lane per fibre (e, p, k)
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    BK1_BATCH_ACCESSORS
                    T reg[nm];
                    for (index_t j = 0; j < nm; ++j) {
                        reg[j] = w1(e, p, j, k);
                    }
                    for (index_t q = 0; q < nq; ++q) {
                        T tmp = 0;
                        for (index_t j = 0; j < nm; ++j) {
                            tmp += reg[j] * B(j, q);
                        }
                        w0(e, p, q, k) = tmp;
                    }
                }
            }
        }

        // step-4 : direction 2, one lane per fibre (e, p, q), fused with
        // step-5 : multiply with weights and determinant of Jacobi
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t q = 0; q < nq; ++q) {
                    BK1_BATCH_ACCESSORS
                    T reg[nm];
                    for (index_t k = 0; k < nm; ++k) {
                        reg[k] = w0(e, p, q, k);
                    }
                    for (index_t r = 0; r < nq; ++r) {
                        T tmp = 0;
                        for (index_t k = 0; k < nm; ++k) {
                            tmp += reg[k] * B(k, r);
                        }
                        w1(e, r, p, q) = tmp * W(e, r, p, q);
                    }
                }
            }
        }

        // Reverse operations

        // step-6 : direction 2, one lane per fibre (e, p, q)
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t q = 0; q < nq; ++q) {
                    BK1_BATCH_ACCESSORS
                    T reg[nq];
                    for (index_t r = 0; r < nq; ++r) {
                        reg[r] = w1(e, r, p, q);
                    }
                    for (index_t k = 0; k < nm; ++k) {
                        T tmp = 0;
                        for (index_t r = 0; r < nq; ++r) {
                            tmp += reg[r] * B(k, r);
                        }
                        w0(e, k, p, q) = tmp;
                    }
                }
            }
        }

        // step-7 : direction 1, one lane per fibre (e, p, k)
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    BK1_BATCH_ACCESSORS
                    T reg[nq];
                    for (index_t q = 0; q < nq; ++q) {
                        reg[q] = w0(e, k, p, q);
                    }
                    for (index_t j = 0; j < nm; ++j) {
                        T tmp = 0;
                        for (index_t q = 0; q < nq; ++q) {
                            tmp += reg[q] * B(j, q);
                        }
                        w1(e, j, p, k) = tmp;
                    }
                }
            }
        }

        // step-8 : direction 0, one lane per fibre (e, j, k), fused with
        // step-9 : the outputs go straight to the element's output block
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    BK1_BATCH_ACCESSORS
                    T reg[nq];
                    for (index_t p = 0; p < nq; ++p) {
                        reg[p] = w1(e, j, p, k);
                    }
                    for (index_t i = 0; i < nm; ++i) {
                        T tmp = 0;
                        for (index_t p = 0; p < nq; ++p) {
                            tmp += reg[p] * B(i, p);
                        }
                        out_e(e, i, j, k) = tmp;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel selection by name, and the configuration line each one reports
// ---------------------------------------------------------------------------
inline const char* kernel_names() { return "elem gang1 gang2 gang3 worker batch"; }

template <typename T, int nq>
bool run_kernel(const std::string& name, const std::size_t nelmt,
                const T* basis, const T* JxW, const T* in, T* out)
{
    if (name == "elem") {
        SumFactorizationElem<T, nq>(nelmt, basis, JxW, in, out);
    } else if (name == "gang1") {
        SumFactorizationGang<T, nq, 1>(nelmt, basis, JxW, in, out);
    } else if (name == "gang2") {
        SumFactorizationGang<T, nq, 2>(nelmt, basis, JxW, in, out);
    } else if (name == "gang3") {
        SumFactorizationGang<T, nq, 3>(nelmt, basis, JxW, in, out);
    } else if (name == "worker") {
        SumFactorizationWorker<T, nq>(nelmt, basis, JxW, in, out);
    } else if (name == "batch") {
        SumFactorizationBatch<T, nq>(nelmt, basis, JxW, in, out);
    } else {
        return false;
    }
    return true;
}

template <int nq>
std::string kernel_config(const std::string& name)
{
    constexpr int epg = batch_elements(nq);
    std::string s = "config: kernel = " + name + " nq = " + std::to_string(nq);
#ifdef BK_CACHE_BASIS
    if (name != "worker") {
        s += " cache(basis)";
    }
#endif
    if (name == "gang1" || name == "gang2" || name == "gang3") {
        const int C = name[4] - '0';
        s += " collapse = " + std::to_string(C)
           + " vector_length = " + std::to_string(gang_vector_length(nq, C));
    } else if (name == "worker") {
        s += " workers = " + std::to_string(worker_count(nq))
           + " vector_length = " + std::to_string(worker_vector_length(nq))
           + " scratch = " + std::to_string(worker_count(nq) * 2 * nq * nq * nq * sizeof(float)) + " bytes";
    } else if (name == "batch") {
        s += " elements/gang = " + std::to_string(epg)
           + " vector_length = " + std::to_string(batch_vector_length(nq, epg))
           + " scratch = " + std::to_string(2 * epg * nq * nq * nq * sizeof(float)) + " bytes";
    }
    return s;
}

} // namespace bk

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
int run_test(const std::string& kernel, const std::size_t nelmt, const int ntests, const bool random)
{
    constexpr int nm = nq - 1;

    const std::array<T, nm * nq> basis = make_test_basis<T, nm, nq>();
    std::vector<T> JxW(nelmt * nq * nq * nq, T(1.0));
    std::vector<T> in (nelmt * nm * nm * nm, T(3.0));
    std::vector<T> out(nelmt * nm * nm * nm);
    if (random) {
        fill_random(JxW, 1u);
        fill_random(in, 2u);
    }

    const std::size_t size_inout = in.size();
    const std::size_t size_JxW   = JxW.size();
    [[maybe_unused]] const std::size_t size_basis = basis.size();   // only the data clause uses it

    const T* d_basis = basis.data();
    const T* d_JxW   = JxW.data();
    const T* d_in    = in.data();
    T*       d_out   = out.data();

    // minimum wall time over ntests repetitions; the kernels are synchronous
    // (no async clause), so the host clock brackets the device execution
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();
    bool known = true;

    #pragma acc data \
        copyin(d_basis[0:size_basis]) \
        copyin(d_JxW[0:size_JxW], d_in[0:size_inout]) \
        copyout(d_out[0:size_inout])
    {
        for (int t = 0; t < ntests && known; ++t) {
            auto start = high_resolution_clock::now();
            known = run_kernel<T, nq>(kernel, nelmt, d_basis, d_JxW, d_in, d_out);
            auto stop = high_resolution_clock::now();
            duration<double> rep_time = stop - start;
            elapsed = std::min(elapsed, rep_time.count());
        }
    }
    if (!known) {
        std::cerr << "unknown kernel BK_KERNEL=" << kernel << " (" << kernel_names() << ")\n";
        return 1;
    }

    // Performance in GDoF/s
    const auto dof_rate = [&](double seconds) {
        return 1.0e-9 * size_inout / seconds;
    };
    // Effective bandwidth in GB/s: read in + write out + read JxW
    const auto byte_rate = [&](double seconds) {
        return 1.0e-9 * sizeof(T) * (2 * size_inout + size_JxW) / seconds;
    };

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";

    std::cout << "norm = " << std::setprecision(9)
              << norm2(out.data(), out.size()) << "\n";

    // The kernel's configuration, on stderr so that stdout stays parseable.
    std::cerr << kernel_config<nq>(kernel) << "\n";
    return 0;
}

// Default element count, the same 2^19 as BK1.cpp.
constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const bool random = random_data_requested();
    const std::string kernel = get_env("BK_KERNEL").value_or("elem");

    // Runtime p -> compile-time nq: one kernel instantiation per supported
    // order, nq = p + 2 (each case label must pair with its literal + 2).
    switch (p) {
        case 1: return run_test<float,  3>(kernel, nelmt, ntests, random);
        case 2: return run_test<float,  4>(kernel, nelmt, ntests, random);
        case 3: return run_test<float,  5>(kernel, nelmt, ntests, random);
        case 4: return run_test<float,  6>(kernel, nelmt, ntests, random);
        case 5: return run_test<float,  7>(kernel, nelmt, ntests, random);
        case 6: return run_test<float,  8>(kernel, nelmt, ntests, random);
        case 7: return run_test<float,  9>(kernel, nelmt, ntests, random);
        case 8: return run_test<float, 10>(kernel, nelmt, ntests, random);
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
}
