// bk_harness.h -- host side of the GPU kernel harnesses (openmp/offload/,
// sycl/, tinytc/): kernel selection, the double-precision references of BK1,
// BK3 and BK5 (host OpenMP), the data layouts and the accuracy check.  No SYCL.

#ifndef BK_HARNESS_H
#define BK_HARNESS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"

namespace bk {

// BK_KERNEL="a b -c": run the variants named, all if none is named, never
// those named with a leading "-" (unset or empty: all).
inline bool selected(const char* name)
{
    static const std::string list = get_env("BK_KERNEL").value_or("");
    bool any = false;
    bool named = false;
    std::istringstream ss(list);
    std::string w;
    while (ss >> w) {
        if (w[0] == '-') {
            if (w.compare(1, std::string::npos, name) == 0) {
                return false;
            }
        } else {
            any = true;
            named = named || (w == name);
        }
    }
    return !any || named;
}

// The first earlier entry of a variant table with the same kernel as entry
// v (a batch clamped to an earlier one's) that is selected, else nullptr.
template <typename V, std::size_t N>
const V* same_kernel(const V (&table)[N], const std::size_t v)
{
    for (std::size_t u = 0; u < v; ++u) {
        if (table[u].fn == table[v].fn && selected(table[u].name)) {
            return &table[u];
        }
    }
    return nullptr;
}

// The references and the check run one element (index) per iteration of an
// OpenMP parallel for (BK_PARALLEL=0: one thread), with an element's work
// arrays on the stack, at most ref_max_nq points per direction.  Each element
// is summed in the same order whatever the thread count.
constexpr int ref_max_nq = 12;

inline bool ref_parallel() { return env_int("BK_PARALLEL", 1) != 0; }

// Largest |out - ref| relative to the largest |ref|: float rounding gives
// ~1e-7, an indexing error ~1.
inline double max_abs(const std::vector<double>& ref)
{
    double big = 0.0;
    for (const double v : ref) {
        big = std::max(big, std::fabs(v));
    }
    return big;
}

inline double rel_maxdiff(const std::vector<float>& out, const std::vector<double>& ref, const double big)
{
    double diff = 0.0;
    [[maybe_unused]] const bool parallel = ref_parallel();
    #pragma omp parallel for if(parallel) reduction(max: diff)
    for (std::size_t x = 0; x < out.size(); ++x) {
        diff = std::max(diff, std::fabs(double(out[x]) - ref[x]));
    }
    return diff / big;
}

inline const char* fail_mark(const double relerr) { return (relerr < 1e-5) ? "" : "  FAIL"; }

// BK1: JxW(e, p, q, r) -> JxW_t(e, r, q, p), for the kernels that read it
// transposed.
inline std::vector<float> transpose_jxw(const std::vector<float>& JxW, const std::size_t nelmt, const int nq)
{
    std::vector<float> JxW_t(JxW.size());
    for (std::size_t e = 0; e < nelmt; ++e) {
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    JxW_t[((e * nq + r) * nq + q) * nq + p] = JxW[((e * nq + p) * nq + q) * nq + r];
                }
            }
        }
    }
    return JxW_t;
}

// BK1 and BK3, one element in double: u(p,q,r) = sum in(i,j,k) B(i,p)
// B(j,q) B(k,r) through a(p,j,k) and b(p,q,k)
inline void ref_interp(const int nq, const std::vector<float>& basis, const float* in, double* a, double* b,
                       double* u)
{
    const int nm = nq - 1;
    for (int p = 0; p < nq; ++p) {
        for (int j = 0; j < nm; ++j) {
            for (int k = 0; k < nm; ++k) {
                double s = 0;
                for (int i = 0; i < nm; ++i) {
                    s += double(in[(i * nm + j) * nm + k]) * basis[i * nq + p];
                }
                a[(p * nm + j) * nm + k] = s;
            }
        }
    }
    for (int p = 0; p < nq; ++p) {
        for (int q = 0; q < nq; ++q) {
            for (int k = 0; k < nm; ++k) {
                double s = 0;
                for (int j = 0; j < nm; ++j) {
                    s += a[(p * nm + j) * nm + k] * basis[j * nq + q];
                }
                b[(p * nq + q) * nm + k] = s;
            }
        }
    }
    for (int p = 0; p < nq; ++p) {
        for (int q = 0; q < nq; ++q) {
            for (int r = 0; r < nq; ++r) {
                double s = 0;
                for (int k = 0; k < nm; ++k) {
                    s += b[(p * nq + q) * nm + k] * basis[k * nq + r];
                }
                u[(p * nq + q) * nq + r] = s;
            }
        }
    }
}

// the transpose: out(i,j,k) = sum u(p,q,r) B(i,p) B(j,q) B(k,r)
inline void ref_interp_back(const int nq, const std::vector<float>& basis, const double* u, double* a, double* b,
                            double* out)
{
    const int nm = nq - 1;
    for (int p = 0; p < nq; ++p) {
        for (int q = 0; q < nq; ++q) {
            for (int k = 0; k < nm; ++k) {
                double s = 0;
                for (int r = 0; r < nq; ++r) {
                    s += u[(p * nq + q) * nq + r] * basis[k * nq + r];
                }
                b[(p * nq + q) * nm + k] = s;
            }
        }
    }
    for (int p = 0; p < nq; ++p) {
        for (int j = 0; j < nm; ++j) {
            for (int k = 0; k < nm; ++k) {
                double s = 0;
                for (int q = 0; q < nq; ++q) {
                    s += b[(p * nq + q) * nm + k] * basis[j * nq + q];
                }
                a[(p * nm + j) * nm + k] = s;
            }
        }
    }
    for (int i = 0; i < nm; ++i) {
        for (int j = 0; j < nm; ++j) {
            for (int k = 0; k < nm; ++k) {
                double s = 0;
                for (int p = 0; p < nq; ++p) {
                    s += a[(p * nm + j) * nm + k] * basis[i * nq + p];
                }
                out[(i * nm + j) * nm + k] = s;
            }
        }
    }
}

// BK1 in double: in(e, i, j, k), basis(i, p), JxW(e, p, q, r) -> out(e, i, j, k).
inline void reference_bk1(const int nq, const std::size_t nelmt, const std::vector<float>& basis,
                          const std::vector<float>& JxW, const std::vector<float>& in, std::vector<double>& out)
{
    assert(nq <= ref_max_nq);
    const int nm3 = (nq - 1) * (nq - 1) * (nq - 1);
    const int nq3 = nq * nq * nq;
    constexpr int box = ref_max_nq * ref_max_nq * ref_max_nq;
    [[maybe_unused]] const bool parallel = ref_parallel();
    #pragma omp parallel for if(parallel)
    for (std::size_t e = 0; e < nelmt; ++e) {
        double a[box];
        double b[box];
        double u[box];
        ref_interp(nq, basis, in.data() + e * nm3, a, b, u);
        for (int x = 0; x < nq3; ++x) {
            u[x] *= JxW[e * nq3 + x];
        }
        ref_interp_back(nq, basis, u, a, b, out.data() + e * nm3);
    }
}

// BK5: the six symmetric factors rr rs rt ss st tt of every point, stored
// interleaved as G6i(e, i, j, s, k), and the full-tensor entry (a, b) ->
// symmetric factor.
constexpr int g6i_index(const int nq, const int i, const int j, const int s, const int k)
{
    return ((i * nq + j) * 6 + s) * nq + k;
}

constexpr int bk5_sym[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};

// G with F = 6 (symmetric) or 9 (full, row-major) factors per point in
// layout 0: factor-major G(f, i, j, k), 1: interleaved G(i, j, f, k) or
// 2: column G(f, k, i, j), built from G6i.
inline std::vector<float> make_bk5_G(const std::vector<float>& G6i, const std::size_t nelmt, const int nq,
                                     const int F, const int layout)
{
    const int nq3 = nq * nq * nq;
    std::vector<float> G(nelmt * F * nq3);
    for (std::size_t e = 0; e < nelmt; ++e) {
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    for (int f = 0; f < F; ++f) {
                        const int s = (F == 6) ? f : bk5_sym[f];
                        int dst = ((f * nq + i) * nq + j) * nq + k;
                        if (layout == 1) {
                            dst = ((i * nq + j) * F + f) * nq + k;
                        } else if (layout == 2) {
                            dst = ((f * nq + k) * nq + i) * nq + j;
                        }
                        G[e * F * nq3 + dst] = G6i[e * 6 * nq3 + g6i_index(nq, i, j, s, k)];
                    }
                }
            }
        }
    }
    return G;
}

// BK5 in double: in(e, i, j, k), D(i, n), G6i -> out(e, i, j, k).
inline void reference_bk5(const int nq, const std::size_t nelmt, const std::vector<float>& D,
                          const std::vector<float>& G6i, const std::vector<float>& in, std::vector<double>& out)
{
    assert(nq <= ref_max_nq);
    const int nq3 = nq * nq * nq;
    [[maybe_unused]] const bool parallel = ref_parallel();
    #pragma omp parallel for if(parallel)
    for (std::size_t e = 0; e < nelmt; ++e) {
        double rr[ref_max_nq * ref_max_nq * ref_max_nq];
        double rs[ref_max_nq * ref_max_nq * ref_max_nq];
        double rt[ref_max_nq * ref_max_nq * ref_max_nq];
        const float* ei = in.data() + e * nq3;
        const float* eG = G6i.data() + e * 6 * nq3;
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    double qr = 0;
                    double qs = 0;
                    double qt = 0;
                    for (int n = 0; n < nq; ++n) {
                        qr += double(ei[(n * nq + j) * nq + k]) * D[i * nq + n];
                        qs += double(ei[(i * nq + n) * nq + k]) * D[j * nq + n];
                        qt += double(ei[(i * nq + j) * nq + n]) * D[k * nq + n];
                    }
                    const auto g = [&](const int f) { return double(eG[g6i_index(nq, i, j, f, k)]); };
                    const int x = (i * nq + j) * nq + k;
                    rr[x] = g(0) * qr + g(1) * qs + g(2) * qt;
                    rs[x] = g(1) * qr + g(3) * qs + g(4) * qt;
                    rt[x] = g(2) * qr + g(4) * qs + g(5) * qt;
                }
            }
        }
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < nq; ++j) {
                for (int k = 0; k < nq; ++k) {
                    double s = 0;
                    for (int n = 0; n < nq; ++n) {
                        s += rr[(n * nq + j) * nq + k] * D[n * nq + i];
                        s += rs[(i * nq + n) * nq + k] * D[n * nq + j];
                        s += rt[(i * nq + j) * nq + n] * D[n * nq + k];
                    }
                    out[e * nq3 + (i * nq + j) * nq + k] = s;
                }
            }
        }
    }
}

// BK3 in double: in(e, i, j, k), basis(i, p), D(n, p), G(e, f, p, q, r)
// (six symmetric factors rr rs rt ss st tt, factor-major) -> out(e, i, j, k):
// interpolate, derivatives with D(n, p), fluxes with G, transposed
// derivatives with D(p, n), interpolate back (openmp/bk3_sumfact.h).
inline void reference_bk3(const int nq, const std::size_t nelmt, const std::vector<float>& basis,
                          const std::vector<float>& D, const std::vector<float>& G,
                          const std::vector<float>& in, std::vector<double>& out)
{
    assert(nq <= ref_max_nq);
    const int nm3 = (nq - 1) * (nq - 1) * (nq - 1);
    const int nq3 = nq * nq * nq;
    constexpr int box = ref_max_nq * ref_max_nq * ref_max_nq;
    [[maybe_unused]] const bool parallel = ref_parallel();
    #pragma omp parallel for if(parallel)
    for (std::size_t e = 0; e < nelmt; ++e) {
        double a[box];
        double b[box];
        double u[box];
        double fr[box];
        double fs[box];
        double ft[box];
        const float* eG = G.data() + e * 6 * nq3;
        const auto Dm = [&](const int n, const int p) { return double(D[n * nq + p]); };
        ref_interp(nq, basis, in.data() + e * nm3, a, b, u);
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    double qr = 0;
                    double qs = 0;
                    double qt = 0;
                    for (int n = 0; n < nq; ++n) {
                        qr += u[(n * nq + q) * nq + r] * Dm(n, p);
                        qs += u[(p * nq + n) * nq + r] * Dm(n, q);
                        qt += u[(p * nq + q) * nq + n] * Dm(n, r);
                    }
                    const int x = (p * nq + q) * nq + r;
                    const auto g = [&](const int f) { return double(eG[f * nq3 + x]); };
                    fr[x] = g(0) * qr + g(1) * qs + g(2) * qt;
                    fs[x] = g(1) * qr + g(3) * qs + g(4) * qt;
                    ft[x] = g(2) * qr + g(4) * qs + g(5) * qt;
                }
            }
        }
        // transposed derivatives -> u
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    double s = 0;
                    for (int n = 0; n < nq; ++n) {
                        s += fr[(n * nq + q) * nq + r] * Dm(p, n);
                        s += fs[(p * nq + n) * nq + r] * Dm(q, n);
                        s += ft[(p * nq + q) * nq + n] * Dm(r, n);
                    }
                    u[(p * nq + q) * nq + r] = s;
                }
            }
        }
        ref_interp_back(nq, basis, u, a, b, out.data() + e * nm3);
    }
}

} // namespace bk

#endif // BK_HARNESS_H
