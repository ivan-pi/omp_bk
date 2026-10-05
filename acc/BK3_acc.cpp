// BK3 (Poisson / stiffness matrix) with OpenACC offloading.
//
// The element loop carries a plain `acc parallel loop`; the inner loop nests
// are left without directives so that `nvc++ -Minfo=acc` shows how the
// compiler schedules them. Every per-element work array is declared inside
// the loop body and is therefore private to its iteration.
//
// Environment: BK_RANDOM=1 fills in and G with seeded random data
// (see acc_common.h); the default is the constant data of BK3.cpp.

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

template <typename T, int nq, int nm = nq - 1, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;

    // The data clauses are no-ops when the arrays are already present
    // (the driver's `acc data` region), and make the kernel usable on its own.
    #pragma acc parallel loop \
        copyin(basis[0:nm*nq], dbasis[0:nq*nq]) \
        copyin(G[0:nelmt*6*nq*nq*nq], in[0:nelmt*nm*nm*nm]) \
        copyout(out[0:nelmt*nm*nm*nm])
    for (std::size_t e = 0; e < nelmt; ++e) {

        // Views onto the basis matrices, constructed inside the compute
        // region so that they hold the device addresses.
        // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
        const ndview<const T, nm, nq> B{basis};
        // D(row, col) == dbasis[row * nq + col], shared by all directions
        const ndview<const T, nq, nq> D{dbasis};

        // Work arrays: five nq^3 boxes; steps address sub-slices of each box.
        // Every step assigns its full output sub-slice, so no zeroing is needed.
        T scratch[5 * nq * nq * nq];
        const ndview<T, nq, nq, nq> wsp0{scratch};
        const ndview<T, nq, nq, nq> wsp1{scratch + 1 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqr {scratch + 2 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqs {scratch + 3 * nq * nq * nq};
        const ndview<T, nq, nq, nq> rqt {scratch + 4 * nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        // G(factor, p, q, r): factor-major within each element
        const ndview<const T, 6, nq, nq, nq> e_G{G + e * (6 * nq * nq * nq)};

        /*
        Interpolate to GL nodes
        */

        // step-1 : copy in -> wsp0 (nm^3 sub-block of the nq^3 box)
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    wsp0(i, j, k) = e_in(i, j, k);
                }
            }
        }

        // step-2 : direction 0
        for (index_t p = 0; p < nq; ++p) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t j = 0; j < nm; ++j) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; ++i) {
                        tmp += wsp0(i, j, k) * B(i, p);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }

        // step-3 : direction 1
        for (index_t q = 0; q < nq; ++q) {
            for (index_t p = 0; p < nq; ++p) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; ++j) {
                        tmp += wsp1(p, j, k) * B(j, q);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }

        // step-4 : direction 2
        for (index_t r = 0; r < nq; ++r) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; ++k) {
                        tmp += wsp0(q, p, k) * B(k, r);
                    }
                    wsp1(p, q, r) = tmp;
                }
            }
        }

        // steps 5-7 : load geometric factors, multiply by D, apply chain rule.
        // The symmetric metric G is applied so that the diagonal factor pairs
        // with the same-direction derivative (Grr*qr, Gss*qs, Gtt*qt), matching
        // BK5. qr/qs/qt are the derivatives along directions 0/1/2.
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {

                    const T Grr = e_G(0, p, q, r);
                    const T Grs = e_G(1, p, q, r);
                    const T Grt = e_G(2, p, q, r);
                    const T Gss = e_G(3, p, q, r);
                    const T Gst = e_G(4, p, q, r);
                    const T Gtt = e_G(5, p, q, r);

                    T qr = 0;
                    T qs = 0;
                    T qt = 0;

                    for (index_t n = 0; n < nq; ++n) {
                        qr += wsp1(n, q, r) * D(n, p);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qs += wsp1(p, n, r) * D(n, q);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qt += wsp1(p, q, n) * D(n, r);
                    }

                    rqr(p, q, r) = Grr * qr + Grs * qs + Grt * qt;
                    rqs(p, q, r) = Grs * qr + Gss * qs + Gst * qt;
                    rqt(p, q, r) = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        // step-8 : compute out vector in GL nodes
        for (index_t p = 0; p < nq; ++p) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t r = 0; r < nq; ++r) {

                    T tmp0 = 0;
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqr(n, q, r) * D(p, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqs(p, n, r) * D(q, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqt(p, q, n) * D(r, n);
                    }

                    wsp1(p, q, r) = tmp0;
                }
            }
        }

        /*
        Interpolate to GLL nodes
        */

        // step-9 : direction 2
        for (index_t k = 0; k < nm; ++k) {
            for (index_t q = 0; q < nq; ++q) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; ++r) {
                        tmp += wsp1(p, q, r) * B(k, r);
                    }
                    wsp0(q, p, k) = tmp;
                }
            }
        }

        // step-10 : direction 1
        for (index_t j = 0; j < nm; ++j) {
            for (index_t k = 0; k < nm; ++k) {
                for (index_t p = 0; p < nq; ++p) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; ++q) {
                        tmp += wsp0(q, p, k) * B(j, q);
                    }
                    wsp1(p, j, k) = tmp;
                }
            }
        }

        // step-11 : direction 0
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; ++p) {
                        tmp += wsp1(p, j, k) * B(i, p);
                    }
                    wsp0(i, j, k) = tmp;
                }
            }
        }

        // step-12 : copy wsp0 -> out
        for (index_t i = 0; i < nm; ++i) {
            for (index_t j = 0; j < nm; ++j) {
                for (index_t k = 0; k < nm; ++k) {
                    e_out(i, j, k) = wsp0(i, j, k);
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

    const std::array<T, nm * nq> basis  = make_test_basis<T, nm, nq>();
    const std::array<T, nq * nq> dbasis = make_test_basis<T, nq, nq>();
    std::vector<T> G (nelmt * 6 * nq * nq * nq, T(2.0));
    std::vector<T> in(nelmt * nm * nm * nm, T(3.0));
    std::vector<T> out(nelmt * nm * nm * nm);
    if (random) {
        fill_random(G, 1u);
        fill_random(in, 2u);
    }

    [[maybe_unused]] const std::size_t size_basis  = basis.size();   // the data clause uses these two
    [[maybe_unused]] const std::size_t size_dbasis = dbasis.size();
    const std::size_t size_G      = G.size();
    const std::size_t size_inout  = in.size();

    const T* d_basis  = basis.data();
    const T* d_dbasis = dbasis.data();
    const T* d_G      = G.data();
    const T* d_in     = in.data();
    T*       d_out    = out.data();

    // minimum wall time over ntests repetitions; the kernel is synchronous
    // (no async clause), so the host clock brackets the device execution
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma acc data \
        copyin(d_basis[0:size_basis], d_dbasis[0:size_dbasis]) \
        copyin(d_G[0:size_G], d_in[0:size_inout]) \
        copyout(d_out[0:size_inout])
    {
        for (int t = 0; t < ntests; ++t) {
            auto start = high_resolution_clock::now();
            SumFactorization<T, nq>(nelmt, d_basis, d_dbasis, d_G, d_in, d_out);
            auto stop = high_resolution_clock::now();
            duration<double> rep_time = stop - start;
            elapsed = std::min(elapsed, rep_time.count());
        }
    }

    // Performance in GDoF/s
    const auto dof_rate = [&](double seconds) {
        return 1.0e-9 * size_inout / seconds;
    };
    // Effective bandwidth in GB/s: read in + write out + read G
    const auto byte_rate = [&](double seconds) {
        return 1.0e-9 * sizeof(T) * (2 * size_inout + size_G) / seconds;
    };

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";

    std::cout << "norm = " << std::setprecision(9)
              << norm2(out.data(), out.size()) << "\n";
}

// Default element count, the same 2^19 as BK3.cpp.
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
