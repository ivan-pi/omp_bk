#ifndef BP_VTK_H
#define BP_VTK_H

// Legacy (ASCII) VTK output of nodal fields.  The GLL nodes of the Cartesian
// mesh form a structured grid of nnode[0] x nnode[1] x nnode[2] points, so
// the file is a STRUCTURED_GRID with explicit point coordinates (the mesh
// may be warped) and one scalar per field in POINT_DATA.  ParaView and VisIt
// read it directly; cells are the trilinear boxes between neighbouring GLL
// nodes, which is adequate for plotting.

#include <array>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bp_mesh.h"

namespace bp {

template <typename T>
using NamedField = std::pair<std::string, const std::vector<T>*>;

// Returns false when the file cannot be written.
template <typename T>
bool write_vtk(const std::string& path, const Mesh& m,
               const std::vector<std::array<double, 3>>& coords,
               const std::vector<NamedField<T>>& fields,
               const std::string& title)
{
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    const int nx = m.nnode[0];
    const int ny = m.nnode[1];
    const int nz = m.nnode[2];
    std::fprintf(f, "# vtk DataFile Version 3.0\n%s\nASCII\nDATASET STRUCTURED_GRID\n",
                 title.c_str());
    std::fprintf(f, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    std::fprintf(f, "POINTS %zu double\n", m.num_nodes());
    // VTK orders structured points with the first dimension fastest
    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const std::array<double, 3>& x = coords[m.node_id(gx, gy, gz)];
                std::fprintf(f, "%.9g %.9g %.9g\n", x[0], x[1], x[2]);
            }
        }
    }
    std::fprintf(f, "POINT_DATA %zu\n", m.num_nodes());
    for (const NamedField<T>& field : fields) {
        std::fprintf(f, "SCALARS %s double 1\nLOOKUP_TABLE default\n", field.first.c_str());
        const std::vector<T>& v = *field.second;
        for (int gz = 0; gz < nz; ++gz) {
            for (int gy = 0; gy < ny; ++gy) {
                for (int gx = 0; gx < nx; ++gx) {
                    std::fprintf(f, "%.9g\n", double(v[m.node_id(gx, gy, gz)]));
                }
            }
        }
    }
    return std::fclose(f) == 0;
}

} // namespace bp

#endif // BP_VTK_H
