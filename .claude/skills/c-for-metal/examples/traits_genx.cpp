// cuSolverDx-style descriptor composition in CM (cmc is C++17).
//
// The pattern of NVIDIA cuSolverDx / cuBLASDx descriptors:
//   using Solver = decltype(Size<16,16,4>() + Precision<float>() + Function<posv>() + ...);
// works in CM because cmc compiles C++17 (__cplusplus == 201703L;
// `-std=` is rejected): variadic templates, decltype, constexpr variable
// templates and if constexpr are all available. Only the kernel-side C++
// restrictions apply (no pointers, references, arrays, dynamic allocation),
// and only a minimal `std` exists (std::is_same, but no is_same_v).
// Run: host_traits traits_genx.spv
#include <cm/cm.h>

namespace cmsolver {
enum class function { getrf, gesv, potrf, posv };
enum class arrangement { col_major, row_major };
enum class fill { lower, upper };

// Operators: empty structs carrying compile-time values.
template <int M, int N, int K = 1> struct Size { static constexpr int m = M, n = N, k = K; };
template <typename T> struct Precision { using type = T; };
template <function F> struct Function { static constexpr function function_value = F; };
template <arrangement A> struct Arrangement { static constexpr arrangement arrangement_value = A; };
template <fill F> struct FillMode { static constexpr fill fill_value = F; };
struct Thread {}; // execution: one CM hardware thread per problem

// The description accumulates operators by inheritance; operator+ only
// computes a TYPE (it is used inside decltype, never executed).
template <typename... Ops> struct Desc : Ops... {};
template <typename... A, typename B> constexpr Desc<A..., B> operator+(Desc<A...>, B) { return {}; }
template <typename A, typename B> constexpr Desc<A, B> operator+(A, B) { return {}; }

// Traits read the composed type.
template <typename D> constexpr int m_of = D::m;
template <typename D> constexpr int k_of = D::k;
template <typename D> using precision_of = typename D::type;
template <typename D> constexpr function function_of = D::function_value;
template <typename D> constexpr fill fill_of = D::fill_value;
// (each operator uses its own member name: two bases both defining `value`
// would make D::value ambiguous)
} // namespace cmsolver

using Solver = decltype(cmsolver::Size<16, 16, 4>() + cmsolver::Precision<float>() +
                        cmsolver::Function<cmsolver::function::posv>() +
                        cmsolver::Arrangement<cmsolver::arrangement::col_major>() +
                        cmsolver::FillMode<cmsolver::fill::lower>() + cmsolver::Thread());

static_assert(cmsolver::m_of<Solver> == 16 && cmsolver::k_of<Solver> == 4, "size");
static_assert(std::is_same<cmsolver::precision_of<Solver>, float>::value, "precision"); // no is_same_v: CM ships a minimal std
static_assert(cmsolver::function_of<Solver> == cmsolver::function::posv, "function");
static_assert(cmsolver::fill_of<Solver> == cmsolver::fill::lower, "fill");

// A kernel configured by the descriptor: register tile sized from it.
template <typename D>
_GENX_ inline void describe(SurfaceIndex out) {
  using T = cmsolver::precision_of<D>;
  matrix<T, cmsolver::m_of<D>, cmsolver::m_of<D>> a = 0; // N x N register tile
  vector<int, 8> info = 0;
  info(0) = cmsolver::m_of<D>;
  info(1) = cmsolver::k_of<D>;
  info(2) = sizeof(T);
  info(3) = int(a.n_rows() * a.n_cols());
  info(4) = __cplusplus / 100;
  write(out, 0, info);
}

extern "C" _GENX_MAIN_ void traits_demo(SurfaceIndex OUT [[type("buffer_t")]]) {
  describe<Solver>(OUT);
}
