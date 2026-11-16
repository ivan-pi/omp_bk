#include <cstddef>
#include <iostream>
#include <array>
#include <vector>

#include "bk_common.h"
#include "bk3_sumfact.h"

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    constexpr int nm = nq - 1;

    const std::array<T, nm * nq> basis  = make_test_basis<T, nm, nq>();
    const std::array<T, nq * nq> dbasis = make_test_basis<T, nq, nq>();
    const std::vector<T> G (nelmt * 6 * nq * nq * nq, T(2.0));
    const std::vector<T> in(nelmt * nm * nm * nm, T(3.0));
    std::vector<T>       out(nelmt * nm * nm * nm);

    const std::size_t size_basis  = basis.size();
    const std::size_t size_dbasis = dbasis.size();
    const std::size_t size_G      = G.size();
    const std::size_t size_inout  = in.size();

    const T* d_basis  = basis.data();
    const T* d_dbasis = dbasis.data();
    const T* d_G      = G.data();
    const T* d_in     = in.data();
    T*       d_out    = out.data();

    // minimum wall time over ntests repetitions, the arrays kept on the device
    double elapsed = 0.0;
    #pragma omp target data \
        map(to: d_basis[:size_basis]) \
        map(to: d_dbasis[:size_dbasis]) \
        map(to: d_G[:size_G], d_in[:size_inout]) \
        map(tofrom: d_out[:size_inout])
    {
        elapsed = time_min(ntests, [&] {
            SumFactorization<T, nq>(nelmt, d_basis, d_dbasis, d_G, d_in, d_out);
        });
    }

    // bytes: read in + write out + read G
    print_rates(nelmt, size_inout, sizeof(T) * (2 * size_inout + size_G), elapsed);
    std::cout << "norm = " << norm2(out.data(), out.size()) << "\n";
}

// <p> [nelmt] [ntests]: polynomial order p = 1..8, nq = p + 2
int main(int argc, char** argv)
{
    const run_args a = parse_args(argc, argv, 2);
    const bool ok = with_order<1, 8>("polynomial order p", a.order, [&](auto P) {
        run_test<float, decltype(P)::value + 2>(a.nelmt, a.ntests);
    });
    return ok ? 0 : 1;
}
