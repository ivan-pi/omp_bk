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
// 15 fp32 flops, cheaper than fp64 at 1/32 or 1/64 of the fp32 rate, and the
// kernel reads and writes the same 8 bytes per value as double.
//
// Two float-float kernels share the arithmetic: SumFactorizationFF runs one
// element per thread with the work arrays in thread-local memory, and
// SumFactorizationFFTeams one element per team with the work arrays shared
// by the team (shared memory on a GPU) and the points of each step spread
// over its threads; the team size is a constexpr of nq (ff_team_size).  On
// the host the teams kernel runs a parallel region per step and element,
// so it is slow there; it is meant for the GPU.
//
// The driver runs the double, float and both float-float kernels on the
// same data and prints one row per kernel (throughput and norm), then the
// relative L2 error of float and of each float-float kernel against the
// double result.  Norms and errors are reduced in native double.  The last
// line is the check: both float-float errors must stay below the
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
                        ff::mul_add(th, tl, wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
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
                        ff::mul_add(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
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
                        ff::mul_add(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
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

// Threads per team of SumFactorizationFFTeams: one per point of the nq^3
// box, the largest output of any step, rounded up to whole warps.  At
// nq = 10 this is the CUDA maximum of 1024; at nq <= 4 it is one or two
// warps, which underfills an SM (several elements per team would be the
// cure).  thread_limit is an upper bound: fewer threads only mean more
// iterations per thread in the collapsed loops.
constexpr int ff_team_size(const int nq)
{
    constexpr int warp = 32;
    return ((nq * nq * nq + warp - 1) / warp) * warp;
}

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
                        ff::mul_add(th, tl, wsp0_hi[WSP(i, j, k)], wsp0_lo[WSP(i, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
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
                        ff::mul_add(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, q, r)], wsp1_lo[WSP(p, q, r)],
                                    B_hi[BAS(k, r)], B_lo[BAS(k, r)]);
                    }
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
                        ff::mul_add(th, tl, wsp0_hi[WSP(q, p, k)], wsp0_lo[WSP(q, p, k)],
                                    B_hi[BAS(j, q)], B_lo[BAS(j, q)]);
                    }
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
                        ff::mul_add(th, tl, wsp1_hi[WSP(p, j, k)], wsp1_lo[WSP(p, j, k)],
                                    B_hi[BAS(i, p)], B_lo[BAS(i, p)]);
                    }
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

// Time `kernel` on the data in precision T, print its row and return its
// output.
template <typename T, int nq>
std::vector<T> run_precision(
    const char* label, const sumfact_kernel<T> kernel,
    const std::size_t nelmt, const int ntests,
    const std::vector<double>& basis64,
    const std::vector<double>& JxW64,
    const std::vector<double>& in64)
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
    return out;
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

    const std::vector<double> ref = run_precision<double, nq>(
        "double", &SumFactorization<double, nq>, nelmt, ntests, basis, JxW, in);
    const double err_float = relative_error(run_precision<float, nq>(
        "float", &SumFactorization<float, nq>, nelmt, ntests, basis, JxW, in), ref);
    const double err_ff = relative_error(run_precision<double, nq>(
        "float-float", &SumFactorizationFF<nq>, nelmt, ntests, basis, JxW, in), ref);
    const double err_teams = relative_error(run_precision<double, nq>(
        "ff-teams", &SumFactorizationFFTeams<nq>, nelmt, ntests, basis, JxW, in), ref);

    const bool ok = (err_ff <= tol) && (err_teams <= tol);
    std::cout << "float        rel. error vs double = " << err_float << "\n"
              << (ok ? "ok" : "FAIL")
              << ": float-float rel. error vs double = " << err_ff
              << ", ff-teams = " << err_teams
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
