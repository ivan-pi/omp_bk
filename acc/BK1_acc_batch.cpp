// BK1 (mass matrix) with OpenACC offloading, batched: each gang holds EPG
// elements in shared memory and every phase is one flat vector loop over the
// fibres of the whole batch, one lane per fibre. A lane loads the inputs
// along the contraction direction into registers (reg[]) and forms every
// output of that line, so each shared-memory value is read once per phase
// instead of once per output. The explicit copy matters: with the loads
// written in place of reg[] in the output loop, nvc++ reloads them for every
// output, and the kernel measured up to 2.3x slower (nq = 8, 1e6 elements).
// This is the structure of the CUDA kernel in
// dealii-X/benchmarks (CEED_BK/include/kernels/BK1/templated_cuda_kernels.cuh)
// written in OpenACC: gang = thread block, `loop vector collapse(3)` over
// (element, free index, free index) = the block-stride loop over fibres with
// the compiler doing the index decomposition, and the barrier nvc++ places
// between consecutive vector loops = __syncthreads().
//
// Layouts. The two work arrays are boxes of nq^3 entries per element in
// every phase; a phase that produces nm values along an index fills the
// first nm. Every phase writes with the lane index as the fastest box index,
// so its stores, and the next phase's loads of the same index, are stride-1
// across lanes; the contracted index is loaded within a lane. Per phase,
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
// Steps 4 and 7 contract the index that the previous phase had fastest, so
// their fibre loads are strided across lanes; one such phase per direction
// sweep is the minimum for a three-direction chain. The input copy, the JxW
// multiply and the output copy are fused into steps 2, 4 and 8, so a batch
// passes through six vector loops and five barriers.
//
// JxW is read in the order step 4 writes: the weight of point (p,q,r) of
// element e sits at e*nq^3 + r*nq^2 + p*nq + q. BK1.cpp uses p*nq^2 + q*nq
// + r; the test data does not depend on the convention.
//
// Environment: BK_RANDOM=1 fills in and JxW with seeded random data
// (see acc_common.h); the default is the constant data of BK1.cpp.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "acc_common.h"

namespace bk {

// Elements per gang: enough fibres (nq^2 per element) to fill about 512
// lanes, capped so that the two work boxes stay within the 48 KB static
// shared-memory allocation of a block. nq = 4: 32 elements; nq = 8: 8;
// nq = 10: 6 (the shared-memory cap, 48000 bytes).
constexpr int elements_per_gang(const int nq)
{
    constexpr int target_lanes = 512;
    constexpr int max_scratch_bytes = 48 * 1024;
    const int by_lanes = target_lanes / (nq * nq);
    const int by_shared = max_scratch_bytes / (2 * nq * nq * nq * int(sizeof(float)));
    const int n = by_lanes < by_shared ? by_lanes : by_shared;
    return n < 1 ? 1 : n;
}

// Vector length: the batch's fibre count rounded up to whole warps. Phases
// with more iterations than lanes (the flat copies, nq^3 per element) are
// strip-mined by the compiler.
constexpr int vector_length(const int nq, const int epg)
{
    constexpr int warp = 32;
    constexpr int max_lanes = 1024;
    const int lanes = ((epg * nq * nq + warp - 1) / warp) * warp;
    return lanes > max_lanes ? max_lanes : lanes;
}

template <typename T, int nq, int EPG = elements_per_gang(nq), typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    static_assert(nq >= 2, "nq = p + 2 with p >= 0");
    static_assert(EPG >= 1, "at least one element per gang");

    constexpr int nm = nq - 1;
    [[maybe_unused]] constexpr int vlen = vector_length(nq, EPG);   // only the pragma uses it
    constexpr int box = nq * nq * nq;             // work array entries per element

    const std::size_t ngang = (nelmt + EPG - 1) / EPG;

    // The data clauses are no-ops when the arrays are already present
    // (the driver's `acc data` region), and make the kernel usable on its own.
    #pragma acc parallel loop gang vector_length(vlen) \
        copyin(basis[0:nm*nq]) \
        copyin(JxW[0:nelmt*nq*nq*nq], in[0:nelmt*nm*nm*nm]) \
        copyout(out[0:nelmt*nm*nm*nm])
    for (std::size_t g = 0; g < ngang; ++g) {

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

        // Accessors, returning the entry by reference through its linear
        // index. w0/w1 are the nq^3 boxes of the work arrays; in_e/out_e the
        // nm^3 blocks of the batch's elements; W the weights of the batch.
        const auto w0 = [&](index_t e, index_t a, index_t b, index_t c) -> T& {
            return wsp0[((e * nq + a) * nq + b) * nq + c];
        };
        const auto w1 = [&](index_t e, index_t a, index_t b, index_t c) -> T& {
            return wsp1[((e * nq + a) * nq + b) * nq + c];
        };
        const auto in_e = [&](index_t e, index_t i, index_t j, index_t k) -> const T& {
            return in[(e0 + e) * (nm * nm * nm) + (i * nm + j) * nm + k];
        };
        const auto out_e = [&](index_t e, index_t i, index_t j, index_t k) -> T& {
            return out[(e0 + e) * (nm * nm * nm) + (i * nm + j) * nm + k];
        };
        const auto W = [&](index_t e, index_t r, index_t p, index_t q) -> const T& {
            return JxW[(e0 + e) * box + (r * nq + p) * nq + q];
        };
        // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
        const auto B = [&](index_t i, index_t p) -> const T& {
            return basis[i * nq + p];
        };

        // step-1 : read the input fibre, fused with
        // step-2 : direction 0, one lane per fibre (e, j, k)
        #pragma acc loop vector collapse(3)
        for (index_t e = 0; e < nb; ++e) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
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

} // namespace bk

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests, const bool random)
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

    // minimum wall time over ntests repetitions; the kernel is synchronous
    // (no async clause), so the host clock brackets the device execution
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma acc data \
        copyin(d_basis[0:size_basis]) \
        copyin(d_JxW[0:size_JxW], d_in[0:size_inout]) \
        copyout(d_out[0:size_inout])
    {
        for (int t = 0; t < ntests; ++t) {
            auto start = high_resolution_clock::now();
            SumFactorization<T, nq>(nelmt, d_basis, d_JxW, d_in, d_out);
            auto stop = high_resolution_clock::now();
            duration<double> rep_time = stop - start;
            elapsed = std::min(elapsed, rep_time.count());
        }
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

    // The batch configuration, on stderr so that stdout stays parseable.
    constexpr int epg = elements_per_gang(nq);
    std::cerr << "batch: nq = " << nq << " elements/gang = " << epg
              << " vector_length = " << vector_length(nq, epg)
              << " scratch = " << 2 * epg * nq * nq * nq * sizeof(T) << " bytes\n";
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

    // Runtime p -> compile-time nq: one kernel instantiation per supported
    // order, nq = p + 2 (each case label must pair with its literal + 2).
    switch (p) {
        case 1: run_test<float,  3>(nelmt, ntests, random); break;
        case 2: run_test<float,  4>(nelmt, ntests, random); break;
        case 3: run_test<float,  5>(nelmt, ntests, random); break;
        case 4: run_test<float,  6>(nelmt, ntests, random); break;
        case 5: run_test<float,  7>(nelmt, ntests, random); break;
        case 6: run_test<float,  8>(nelmt, ntests, random); break;
        case 7: run_test<float,  9>(nelmt, ntests, random); break;
        case 8: run_test<float, 10>(nelmt, ntests, random); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return 0;
}
