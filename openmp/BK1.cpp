#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <cstddef>

#include "bk_common.h"
#include "bk1_sumfact.h"

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    constexpr int nm = nq - 1;

    const std::array<T, nm * nq> basis = make_test_basis<T, nm, nq>();
    const std::vector<T> JxW(nelmt * nq * nq * nq, T(1.0));
    const std::vector<T> in (nelmt * nm * nm * nm, T(3.0));
    std::vector<T>       out(nelmt * nm * nm * nm);

    // minimum wall time over ntests repetitions
    const double elapsed = time_sumfact<T, nq>(
        nelmt, ntests, basis.data(), JxW.data(), in.data(), out.data());

    // bytes: read in + write out + read JxW
    print_rates(nelmt, in.size(), sizeof(T) * (2 * in.size() + JxW.size()), elapsed);
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
