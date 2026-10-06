#ifndef BP_MESH_H
#define BP_MESH_H

// Cartesian hexahedral mesh of the unit cube and its element restriction.
//
// Terminology follows libCEED: the L-vector holds one value per global
// degree of freedom (node), the E-vector holds one value per element and
// local node, so shared nodes appear several times.  The restriction maps
// between the two:
//
//     E = P   L     gather    (restrict):  E[s] = L[e_to_l[s]]
//     L = P^T E     scatter   (prolong):   L[g] = sum of E[s] over the
//                                          entries s of element nodes
//                                          lying on global node g
//
// The maps are plain index arrays, so the same solver would work on an
// unstructured mesh; only this file knows that the mesh is Cartesian.

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "bp_basis.h"

namespace bp {

struct Mesh {
    int p = 0;                 // polynomial order
    int nm = 0;                // nodes per direction and element, p + 1
    std::array<int, 3> nelem{};   // elements per direction
    std::array<int, 3> nnode{};   // global nodes per direction, nelem * p + 1
    double warp = 0.0;         // amplitude of the smooth mesh deformation

    std::size_t num_elements() const
    {
        return std::size_t(nelem[0]) * nelem[1] * nelem[2];
    }
    std::size_t num_nodes() const
    {
        return std::size_t(nnode[0]) * nnode[1] * nnode[2];
    }
    std::size_t num_evec() const { return num_elements() * nm * nm * nm; }

    // Global node id: z (direction 2) runs fastest, like the local (i, j, k)
    // index of the kernels where k is innermost.
    std::size_t node_id(const int gx, const int gy, const int gz) const
    {
        return (std::size_t(gx) * nnode[1] + gy) * nnode[2] + gz;
    }
    std::size_t element_id(const int ex, const int ey, const int ez) const
    {
        return (std::size_t(ex) * nelem[1] + ey) * nelem[2] + ez;
    }

    // Smooth deformation of the unit cube onto itself: every point moves by
    // warp * sin(pi x) sin(pi y) sin(pi z) along (1, 1, 1).  The boundary
    // stays in place, so the Dirichlet conditions and the manufactured
    // solutions are unaffected; the metric factors become non-constant.
    std::array<double, 3> physical(const std::array<double, 3>& X) const
    {
        const double s = warp * std::sin(M_PI * X[0]) * std::sin(M_PI * X[1])
                              * std::sin(M_PI * X[2]);
        return {X[0] + s, X[1] + s, X[2] + s};
    }

    // Jacobian d(physical)/dX of the deformation.
    std::array<std::array<double, 3>, 3> physical_jacobian(const std::array<double, 3>& X) const
    {
        const double sx = std::sin(M_PI * X[0]);
        const double sy = std::sin(M_PI * X[1]);
        const double sz = std::sin(M_PI * X[2]);
        const double cx = std::cos(M_PI * X[0]);
        const double cy = std::cos(M_PI * X[1]);
        const double cz = std::cos(M_PI * X[2]);
        const std::array<double, 3> ds = {M_PI * warp * cx * sy * sz,
                                          M_PI * warp * sx * cy * sz,
                                          M_PI * warp * sx * sy * cz};
        std::array<std::array<double, 3>, 3> J{};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                J[a][b] = (a == b ? 1.0 : 0.0) + ds[b];
            }
        }
        return J;
    }

    // dX/dxi of the undeformed element: half its size in direction d.
    double half_h(const int d) const { return 0.5 / nelem[d]; }

    // Undeformed coordinate of a reference point xi in [-1, 1] of element e.
    std::array<double, 3> undeformed(const std::array<int, 3>& e,
                                     const std::array<double, 3>& xi) const
    {
        std::array<double, 3> X{};
        for (int d = 0; d < 3; ++d) {
            X[d] = half_h(d) * (2.0 * e[d] + 1.0 + xi[d]);
        }
        return X;
    }
};

inline Mesh make_mesh(const int p, const std::array<int, 3>& nelem, const double warp)
{
    assert(p >= 1);
    Mesh m;
    m.p = p;
    m.nm = p + 1;
    m.nelem = nelem;
    m.warp = warp;
    for (int d = 0; d < 3; ++d) {
        assert(nelem[d] >= 1);
        m.nnode[d] = nelem[d] * p + 1;
    }
    // the ability to warp the mesh without folding it: |d s| < 1 / 3
    assert(std::fabs(warp) * M_PI * std::sqrt(3.0) < 1.0);
    return m;
}

// Element restriction as index arrays (int: the mini-app targets sizes well
// below 2^31 entries, asserted in make_restriction).
struct Restriction {
    std::vector<int> e_to_l;      // E-vector entry -> L-vector entry
    std::vector<int> l_offsets;   // CSR offsets of the transpose, size nL + 1
    std::vector<int> l_to_e;      // E-vector entries of each L-vector entry
};

inline Restriction make_restriction(const Mesh& m)
{
    const std::size_t nL = m.num_nodes();
    const std::size_t nE = m.num_evec();
    assert(nL < std::size_t(1) << 31);
    assert(nE < std::size_t(1) << 31);
    Restriction r;
    r.e_to_l.resize(nE);
    std::vector<int> count(nL, 0);
    const int nm = m.nm;
    for (int ex = 0; ex < m.nelem[0]; ++ex) {
        for (int ey = 0; ey < m.nelem[1]; ++ey) {
            for (int ez = 0; ez < m.nelem[2]; ++ez) {
                const std::size_t e = m.element_id(ex, ey, ez);
                for (int i = 0; i < nm; ++i) {
                    for (int j = 0; j < nm; ++j) {
                        for (int k = 0; k < nm; ++k) {
                            const std::size_t s = ((e * nm + i) * nm + j) * nm + k;
                            const std::size_t g = m.node_id(ex * m.p + i, ey * m.p + j, ez * m.p + k);
                            r.e_to_l[s] = static_cast<int>(g);
                            ++count[g];
                        }
                    }
                }
            }
        }
    }
    // transpose map in CSR form, deterministic scatter order
    r.l_offsets.resize(nL + 1);
    r.l_offsets[0] = 0;
    for (std::size_t g = 0; g < nL; ++g) {
        r.l_offsets[g + 1] = r.l_offsets[g] + count[g];
    }
    r.l_to_e.resize(nE);
    std::vector<int> fill(r.l_offsets.begin(), r.l_offsets.end() - 1);
    for (std::size_t s = 0; s < nE; ++s) {
        const int g = r.e_to_l[s];
        r.l_to_e[fill[g]++] = static_cast<int>(s);
    }
    return r;
}

// Physical coordinates of every global node (L-vector layout).
inline std::vector<std::array<double, 3>> node_coordinates(const Mesh& m, const Basis1D& b)
{
    std::vector<std::array<double, 3>> x(m.num_nodes());
    for (int gx = 0; gx < m.nnode[0]; ++gx) {
        for (int gy = 0; gy < m.nnode[1]; ++gy) {
            for (int gz = 0; gz < m.nnode[2]; ++gz) {
                const std::array<int, 3> g = {gx, gy, gz};
                std::array<int, 3> e{};
                std::array<double, 3> xi{};
                for (int d = 0; d < 3; ++d) {
                    // the last node of the last element belongs to no next element
                    e[d] = (g[d] == m.nnode[d] - 1) ? m.nelem[d] - 1 : g[d] / m.p;
                    xi[d] = b.nodes.x[g[d] - e[d] * m.p];
                }
                x[m.node_id(gx, gy, gz)] = m.physical(m.undeformed(e, xi));
            }
        }
    }
    return x;
}

// 1 on interior nodes, 0 on the boundary of the cube (homogeneous Dirichlet).
template <typename T>
std::vector<T> dirichlet_mask(const Mesh& m)
{
    std::vector<T> mask(m.num_nodes(), T(1));
    for (int gx = 0; gx < m.nnode[0]; ++gx) {
        for (int gy = 0; gy < m.nnode[1]; ++gy) {
            for (int gz = 0; gz < m.nnode[2]; ++gz) {
                const bool boundary = gx == 0 || gx == m.nnode[0] - 1
                                   || gy == 0 || gy == m.nnode[1] - 1
                                   || gz == 0 || gz == m.nnode[2] - 1;
                if (boundary) {
                    mask[m.node_id(gx, gy, gz)] = T(0);
                }
            }
        }
    }
    return mask;
}

} // namespace bp

#endif // BP_MESH_H
