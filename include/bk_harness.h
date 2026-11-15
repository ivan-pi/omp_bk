// bk_harness.h -- host side of the GPU kernel harnesses (openmp/offload/,
// sycl/, tinytc/): kernel selection, the double-precision references (host
// OpenMP), the BK1 and BK5 data layouts and the accuracy check.  No SYCL.

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

// BK_KERNEL="a b c": run only the variants named (unset or empty: all).
inline bool selected(const char* name)
{
    static const std::string list = get_env("BK_KERNEL").value_or("");
    if (list.empty()) {
        return true;
    }
    std::istringstream ss(list);
    std::string w;
    while (ss >> w) {
        if (w == name) {
            return true;
        }
    }
    return false;
}

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

// The references run one element per iteration of an OpenMP parallel for
// (BK_PARALLEL=0: one thread), with the work arrays of an element on the
// stack, at most ref_max_nq points per direction.  Each element is summed in
// the same order whatever the thread count.
constexpr int ref_max_nq = 12;

inline bool ref_parallel() { return env_int("BK_PARALLEL", 1) != 0; }

// BK1 in double: in(e, i, j, k), basis(i, p), JxW(e, p, q, r) -> out(e, i, j, k).
inline void reference_bk1(const int nq, const std::size_t nelmt, const std::vector<float>& basis,
                          const std::vector<float>& JxW, const std::vector<float>& in, std::vector<double>& out)
{
    assert(nq <= ref_max_nq);
    const int nm = nq - 1;
    [[maybe_unused]] const bool parallel = ref_parallel();
    #pragma omp parallel for if(parallel)
    for (std::size_t e = 0; e < nelmt; ++e) {
        double w0[ref_max_nq * ref_max_nq * ref_max_nq];
        double w1[ref_max_nq * ref_max_nq * ref_max_nq];
        double w2[ref_max_nq * ref_max_nq * ref_max_nq];
        const float* ei = in.data() + e * nm * nm * nm;
        const float* ej = JxW.data() + e * nq * nq * nq;
        // forward: w0(p,j,k), w1(p,q,k), w2(p,q,r) * JxW(p,q,r)
        for (int p = 0; p < nq; ++p) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int i = 0; i < nm; ++i) {
                        s += double(ei[(i * nm + j) * nm + k]) * basis[i * nq + p];
                    }
                    w0[(p * nm + j) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int j = 0; j < nm; ++j) {
                        s += w0[(p * nm + j) * nm + k] * basis[j * nq + q];
                    }
                    w1[(p * nq + q) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int r = 0; r < nq; ++r) {
                    double s = 0;
                    for (int k = 0; k < nm; ++k) {
                        s += w1[(p * nq + q) * nm + k] * basis[k * nq + r];
                    }
                    w2[(p * nq + q) * nq + r] = s * ej[(p * nq + q) * nq + r];
                }
            }
        }
        // backward: w1(p,q,k), w0(p,j,k), out(i,j,k)
        for (int p = 0; p < nq; ++p) {
            for (int q = 0; q < nq; ++q) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int r = 0; r < nq; ++r) {
                        s += w2[(p * nq + q) * nq + r] * basis[k * nq + r];
                    }
                    w1[(p * nq + q) * nm + k] = s;
                }
            }
        }
        for (int p = 0; p < nq; ++p) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int q = 0; q < nq; ++q) {
                        s += w1[(p * nq + q) * nm + k] * basis[j * nq + q];
                    }
                    w0[(p * nm + j) * nm + k] = s;
                }
            }
        }
        for (int i = 0; i < nm; ++i) {
            for (int j = 0; j < nm; ++j) {
                for (int k = 0; k < nm; ++k) {
                    double s = 0;
                    for (int p = 0; p < nq; ++p) {
                        s += w0[(p * nm + j) * nm + k] * basis[i * nq + p];
                    }
                    out[e * nm * nm * nm + (i * nm + j) * nm + k] = s;
                }
            }
        }
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

} // namespace bk

#endif // BK_HARNESS_H
