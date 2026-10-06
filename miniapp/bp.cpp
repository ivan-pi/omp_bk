// bp -- a mini-app that solves the CEED bake-off problems with the BK
// kernels of this repository: the mass solve (BP1), the Poisson problem
// (BP3 with Gauss-Legendre quadrature, BP5 collocated on GLL points) and an
// implicit diffusion (heat) equation that needs both the mass and the
// stiffness operator in every conjugate-gradient iteration.
//
//   ./bp mass    -p 3 -n 8                 M u = M f,  u = f      (BK1)
//   ./bp poisson -p 3 -n 8                 K u = M f              (BK3, BK1)
//   ./bp poisson -p 3 -n 8 --gll           K u = M f, collocated  (BK5)
//   ./bp heat    -p 3 -n 8 --dt 1e-3 --steps 10
//           (M + theta dt K) u^{n+1} = (M - (1 - theta) dt K) u^n   (BK1, BK3)
//   ./bp transport -p 3 -n 8 --velocity 1,0,0 --dt 5e-3 --steps 80
//           M (u^{n+1} - u^n) = r_TG(u^n)   Taylor-Galerkin   (tg_kernel.h, BK1)
//           with --tg3: (M + dt^2/6 K_e) (u^{n+1} - u^n) = r_TG(u^n)  (+ BK3)
//
// The manufactured solution on the unit cube with homogeneous Dirichlet
// conditions is u(x, t) = exp(-3 pi^2 t) sin(pi x) sin(pi y) sin(pi z); the
// transport problem moves a Gaussian bump exp(-|x - x0 - e t|^2 / 2 sigma^2)
// that stays away from the walls.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "bp_basis.h"
#include "bp_mesh.h"
#include "bp_geometry.h"
#include "bp_backend.h"
#include "bp_operator.h"
#include "bp_solver.h"
#include "bp_vtk.h"

#ifndef BP_REAL
#define BP_REAL double
#endif
using real = BP_REAL;

namespace {

enum class Problem { mass, poisson, heat, transport };

struct Options {
    Problem problem = Problem::poisson;
    int p = 3;
    std::array<int, 3> nelem = {8, 8, 8};
    bool collocated = false;
    bool jacobi = false;
    bool atomic = false;
    bool fused = false;
    double warp = 0.0;
    double rtol = 1e-8;
    int maxit = 1000;
    double dt = 1e-3;
    int steps = 10;
    double theta = 1.0;
    std::array<double, 3> velocity = {1.0, 0.0, 0.0};   // transport
    double sigma = 0.08;                                 // width of the bump
    bool tg3 = false;                                    // M + dt^2/6 K_e
    std::string vtk;   // output base name; empty: no output
};

const char* problem_name(const Problem p)
{
    switch (p) {
        case Problem::mass: return "mass";
        case Problem::poisson: return "poisson";
        case Problem::heat: return "heat";
        case Problem::transport: return "transport";
    }
    return "?";
}

void usage()
{
    std::cerr <<
        "usage: bp [mass|poisson|heat|transport] [options]\n"
        "  -p <order>        polynomial order 1..8 (1..7 with --gll), default 3\n"
        "  -n <n> | <nx,ny,nz>  elements per direction, default 8\n"
        "  --gll             collocated GLL quadrature (BK5) instead of GL (BK1/BK3)\n"
        "  --pc jacobi|none  diagonal preconditioner, default none\n"
        "  --atomic          scatter with atomic adds instead of the transpose map\n"
        "  --fused           gather on the fly: no E-vector, atomic scatter inside the kernels\n"
        "  --warp <a>        deform the mesh, |a| < 0.18, default 0\n"
        "  --tol <rtol>      CG relative residual tolerance, default 1e-8;\n"
        "                    0 runs exactly --maxit iterations (benchmark mode)\n"
        "  --maxit <n>       CG iteration limit, default 1000\n"
        "  --dt <dt> --steps <n> --theta <t>   heat: time step, number of steps,\n"
        "                    theta = 1 backward Euler (default), 0.5 Crank-Nicolson\n"
        "  --velocity <ex,ey,ez> --sigma <s> --tg3   transport: constant velocity\n"
        "                    (default 1,0,0), bump width (0.08), third-order Taylor-Galerkin\n"
        "  --vtk <base>      write u, u_exact and the error to <base>.vtk (legacy ASCII\n"
        "                    structured grid); heat writes <base>_<step>.vtk per step\n";
}

bool parse(const int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&](double& v) {
            if (i + 1 >= argc) {
                return false;
            }
            v = std::atof(argv[++i]);
            return true;
        };
        if (a == "mass") {
            o.problem = Problem::mass;
        } else if (a == "poisson") {
            o.problem = Problem::poisson;
        } else if (a == "heat") {
            o.problem = Problem::heat;
        } else if (a == "transport") {
            o.problem = Problem::transport;
        } else if (a == "--tg3") {
            o.tg3 = true;
        } else if (a == "--velocity" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%lf,%lf,%lf", &o.velocity[0], &o.velocity[1], &o.velocity[2]) != 3) {
                return false;
            }
        } else if (a == "--gll") {
            o.collocated = true;
        } else if (a == "--atomic") {
            o.atomic = true;
        } else if (a == "--fused") {
            o.fused = true;
        } else if (a == "--pc" && i + 1 < argc) {
            const std::string pc = argv[++i];
            o.jacobi = (pc == "jacobi");
            if (!o.jacobi && pc != "none") {
                return false;
            }
        } else if (a == "--vtk" && i + 1 < argc) {
            o.vtk = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            const std::string n = argv[++i];
            if (std::sscanf(n.c_str(), "%d,%d,%d", &o.nelem[0], &o.nelem[1], &o.nelem[2]) == 1) {
                o.nelem[1] = o.nelem[0];
                o.nelem[2] = o.nelem[0];
            }
        } else {
            double v = 0.0;
            if (!value(v)) {
                return false;
            }
            if (a == "-p") {
                o.p = static_cast<int>(v);
            } else if (a == "--warp") {
                o.warp = v;
            } else if (a == "--tol") {
                o.rtol = v;
            } else if (a == "--maxit") {
                o.maxit = static_cast<int>(v);
            } else if (a == "--dt") {
                o.dt = v;
            } else if (a == "--steps") {
                o.steps = static_cast<int>(v);
            } else if (a == "--theta") {
                o.theta = v;
            } else if (a == "--sigma") {
                o.sigma = v;
            } else {
                return false;
            }
        }
    }
    return true;
}

// The manufactured solutions: the decaying sine mode, or for transport a
// Gaussian bump centred at x0 + e t with x0 chosen so that the bump crosses
// the centre of the cube half-way through the run.
struct Exact {
    Problem problem = Problem::poisson;
    std::array<double, 3> velocity{};
    std::array<double, 3> x0{};
    double sigma = 0.0;

    double operator()(const std::array<double, 3>& x, const double t) const
    {
        if (problem == Problem::transport) {
            double r2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                const double dx = x[d] - x0[d] - velocity[d] * t;
                r2 += dx * dx;
            }
            return std::exp(-0.5 * r2 / (sigma * sigma));
        }
        return std::exp(-3.0 * M_PI * M_PI * t)
             * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
    }
};

Exact make_exact(const Options& o)
{
    Exact ex;
    ex.problem = o.problem;
    ex.velocity = o.velocity;
    ex.sigma = o.sigma;
    const double t_end = o.dt * o.steps;
    for (int d = 0; d < 3; ++d) {
        ex.x0[d] = 0.5 - 0.5 * t_end * o.velocity[d];
    }
    return ex;
}

template <typename T>
double max_abs(const std::vector<T>& v)
{
    double m = 0.0;
    for (const T x : v) {
        m = std::max(m, std::fabs(double(x)));
    }
    return m;
}

// Writes the solution x, the exact solution at time t and their difference.
template <typename T>
bool write_fields(const std::string& path, const bp::Mesh& mesh,
                  const std::vector<std::array<double, 3>>& coords,
                  const std::vector<T>& x, const Exact& exact, const double t)
{
    std::vector<T> ex(x.size());
    std::vector<T> err(x.size());
    for (std::size_t g = 0; g < x.size(); ++g) {
        ex[g] = T(exact(coords[g], t));
        err[g] = x[g] - ex[g];
    }
    const std::vector<bp::NamedField<T>> fields = {{"u", &x}, {"u_exact", &ex}, {"error", &err}};
    const bool ok = bp::write_vtk(path, mesh, coords, fields, "bp solution, t = " + std::to_string(t));
    if (!ok) {
        std::fprintf(stderr, "could not write %s\n", path.c_str());
    }
    return ok;
}

std::string step_file(const std::string& base, const int step)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "_%04d", step);
    return base + buf + ".vtk";
}

template <typename T, int nq, bool collocated>
int run(const Options& o)
{
    using Op = bp::Operator<T, nq, collocated>;
    constexpr int nm = Op::nm;
    static_assert(nm == (collocated ? nq : nq - 1), "node count of the family");

    // ---- setup on the host -------------------------------------------------
    const bool transport = o.problem == Problem::transport;
    if (transport && (collocated || o.warp != 0.0)) {
        std::cerr << "transport needs the Gauss-Legendre family on the undeformed mesh\n";
        return 1;
    }
    const bp::Basis1D basis = bp::make_basis(o.p, collocated);
    const bp::Mesh mesh = bp::make_mesh(o.p, o.nelem, o.warp);
    const bp::Restriction R = bp::make_restriction(mesh);
    const auto coords = bp::node_coordinates(mesh, basis);
    // transport: the stiffness is the directional one, int (e.grad u)(e.grad v)
    const bp::Geometry<T> geo = bp::make_geometry<T>(mesh, basis, transport ? &o.velocity : nullptr);
    const Exact exact = make_exact(o);
    const std::size_t nelmt = mesh.num_elements();
    const std::size_t nL = mesh.num_nodes();
    const std::size_t nE = mesh.num_evec();
    assert(basis.nq == nq && basis.nm == nm);

    const std::vector<T> B = bp::to_precision<T>(basis.B);
    const std::vector<T> D = bp::to_precision<T>(collocated ? basis.Dq_T : basis.Dq);
    const std::vector<T> w1d = bp::to_precision<T>(basis.quad.w);
    const std::vector<T> mask = bp::dirichlet_mask<T>(mesh);
    const bool use_mask = o.problem != Problem::mass;

    // Operator coefficients of the linear system A = cM M + cK K
    T cM = 0;
    T cK = 0;
    if (o.problem == Problem::mass) {
        cM = 1;
    } else if (o.problem == Problem::poisson) {
        cK = 1;
    } else if (o.problem == Problem::heat) {
        cM = 1;
        cK = T(o.theta * o.dt);
    } else {
        cM = 1;
        cK = o.tg3 ? T(o.dt * o.dt / 6.0) : T(0);
    }
    // Taylor-Galerkin coefficients: r = int phi a (e.grad u) + int (e.grad phi) c (e.grad u)
    const T tg_a = T(-o.dt);
    const T tg_c = T(-0.5 * o.dt * o.dt);

    // Jacobi: assemble the element diagonals into an L-vector and invert
    std::vector<T> dinv(nL, T(0));
    for (std::size_t s = 0; s < nE; ++s) {
        dinv[R.e_to_l[s]] += cM * geo.diag_mass[s] + cK * geo.diag_stiff[s];
    }
    for (std::size_t g = 0; g < nL; ++g) {
        dinv[g] = T(1) / dinv[g];
    }

    // L-vectors: solution, right-hand side, exact solution at the nodes
    std::vector<T> x(nL, T(0));
    std::vector<T> b(nL, T(0));
    std::vector<T> u_exact(nL);
    std::vector<T> f(nL);
    const bool time_dependent = o.problem == Problem::heat || transport;
    const double t_end = time_dependent ? o.dt * o.steps : 0.0;
    for (std::size_t g = 0; g < nL; ++g) {
        u_exact[g] = T(exact(coords[g], t_end));
        f[g] = T(exact(coords[g], 0.0));
        if (o.problem == Problem::poisson) {
            f[g] *= T(3.0 * M_PI * M_PI);    // -Laplace u = f
        }
    }
    if (time_dependent) {
        x = f;                               // initial condition u(x, 0)
    }

    // work space
    std::vector<T> r(nL);
    std::vector<T> z(nL);
    std::vector<T> pv(nL);
    std::vector<T> Ap(nL);
    std::vector<T> delta(nL);               // transport: the update per step
    std::vector<T> e_in(nE);
    std::vector<T> e_mass(nE);
    std::vector<T> e_stiff(nE);

    Op A;
    A.nelmt = nelmt;
    A.nL = nL;
    A.nE = nE;
    A.e_to_l = R.e_to_l.data();
    A.l_offsets = R.l_offsets.data();
    A.l_to_e = R.l_to_e.data();
    A.atomic_scatter = o.atomic;
    A.fused = o.fused;
    A.B = B.data();
    A.D = D.data();
    A.JxW = geo.JxW.data();
    A.G = geo.G.data();
    A.mask = use_mask ? mask.data() : nullptr;
    A.w1d = w1d.data();
    A.detJ = T(1.0 / (8.0 * o.nelem[0] * o.nelem[1] * o.nelem[2]));   // prod h_d / 2
    for (int d = 0; d < 3; ++d) {
        A.et[d] = T(2.0 * o.nelem[d] * o.velocity[d]);                   // e_d / (h_d / 2)
    }
    A.e_in = e_in.data();
    A.e_mass = e_mass.data();
    A.e_stiff = e_stiff.data();

    // raw pointers for the map clauses
    const int* d_e_to_l = R.e_to_l.data();
    const int* d_l_offsets = R.l_offsets.data();
    const int* d_l_to_e = R.l_to_e.data();
    const T* d_B = B.data();
    const T* d_D = D.data();
    const T* d_JxW = geo.JxW.data();
    const T* d_G = geo.G.data();
    const T* d_mask = mask.data();
    const T* d_dinv = dinv.data();
    const T* d_w1d = w1d.data();
    T* d_x = x.data();
    T* d_delta = delta.data();
    T* d_b = b.data();
    T* d_f = f.data();
    T* d_r = r.data();
    T* d_z = z.data();
    T* d_p = pv.data();
    T* d_Ap = Ap.data();
    T* d_e_in = e_in.data();
    T* d_e_mass = e_mass.data();
    T* d_e_stiff = e_stiff.data();
    const bp::CGWorkspace<T> w{d_r, d_z, d_p, d_Ap};

    std::printf("bp %s: p = %d, nq = %d (%s), %d x %d x %d elements, %zu nodes, %zu E-vector entries\n",
                problem_name(o.problem), o.p, nq, collocated ? "collocated GLL, BK5" : "Gauss-Legendre, BK1/BK3",
                o.nelem[0], o.nelem[1], o.nelem[2], nL, nE);
    std::printf("precision %zu bytes, preconditioner %s, operator %s, warp %g\n",
                sizeof(T), o.jacobi ? "jacobi" : "none",
                o.fused ? "fused (gather on the fly, atomic scatter)"
                        : o.atomic ? "E-vector, atomic scatter"
                                   : "E-vector, transpose-map scatter",
                o.warp);
    if (transport) {
        std::printf("velocity (%g, %g, %g), sigma %g, %s Taylor-Galerkin, dt %g, %d steps\n",
                    o.velocity[0], o.velocity[1], o.velocity[2], o.sigma,
                    o.tg3 ? "third-order (M + dt^2/6 K_e)" : "second-order (M)", o.dt, o.steps);
    }

    double volume = 0.0;
    double tg_linear = 0.0;
    double k_const = 0.0;
    double k_scale = 0.0;
    double solve_seconds = 0.0;
    long total_iterations = 0;
    int solves = 0;
    const bool fixed_iterations = o.rtol <= 0.0;
    bool converged = true;
    double err_max = 0.0;
    double err_M = 0.0;
    bool vtk_ok = true;

    // ---- everything below runs with the arrays resident on the device ------
    #pragma omp target data \
        map(to: d_e_to_l[:nE], d_l_offsets[:nL + 1], d_l_to_e[:nE]) \
        map(to: d_B[:B.size()], d_D[:D.size()], d_JxW[:geo.JxW.size()], d_G[:geo.G.size()]) \
        map(to: d_mask[:nL], d_dinv[:nL], d_f[:nL], d_w1d[:nq]) \
        map(tofrom: d_x[:nL]) \
        map(alloc: d_b[:nL], d_r[:nL], d_z[:nL], d_p[:nL], d_Ap[:nL], d_delta[:nL]) \
        map(alloc: d_e_in[:nE], d_e_mass[:nE], d_e_stiff[:nE])
    {
        // sanity checks on the unmasked operators: 1^T M 1 is the volume of
        // the domain and K 1 vanishes, since constants have no gradient
        {
            const T* saved_mask = A.mask;
            A.mask = nullptr;
            bp::backend::fill(nL, T(1), d_r);
            A.apply(T(1), T(0), d_r, d_Ap);
            volume = double(bp::backend::dot(nL, d_r, d_Ap));
            A.apply(T(0), T(1), d_r, d_Ap);
            bp::backend::from_device(nL, d_Ap);
            k_const = max_abs(Ap);
            A.apply(T(0), T(1), d_f, d_Ap);
            bp::backend::from_device(nL, d_Ap);
            k_scale = max_abs(Ap);
            if constexpr (!collocated) {
            if (transport) {
                // the Taylor-Galerkin kernel on the linear function u = e.x:
                // e.grad u = |e|^2, so on the interior nodes r = a |e|^2 M 1
                // (the flux term integrates to a wall contribution only)
                double e2 = 0.0;
                for (std::size_t g = 0; g < nL; ++g) {
                    r[g] = T(o.velocity[0] * coords[g][0] + o.velocity[1] * coords[g][1]
                           + o.velocity[2] * coords[g][2]);
                }
                for (int d = 0; d < 3; ++d) {
                    e2 += o.velocity[d] * o.velocity[d];
                }
                bp::backend::to_device(nL, d_r);
                A.mask = saved_mask;
                A.advect(tg_a, tg_c, d_r, d_Ap);              // masked: interior rows
                bp::backend::fill(nL, T(1), d_r);
                A.apply(T(1), T(0), d_r, d_b);                // masked: a |e|^2 M 1
                bp::backend::axpby(nL, T(-tg_a * e2), d_b, T(1), d_Ap);
                bp::backend::from_device(nL, d_Ap);
                bp::backend::from_device(nL, d_b);
                tg_linear = max_abs(Ap) / (std::fabs(tg_a * e2) * max_abs(b));
            }
            }
            A.mask = saved_mask;
        }

        // right-hand side and solve(s)
        const T* pc = o.jacobi ? d_dinv : nullptr;
        const int nsolve = time_dependent ? o.steps : 1;
        if (time_dependent && !o.vtk.empty()) {
            vtk_ok = write_fields(step_file(o.vtk, 0), mesh, coords, x, exact, 0.0);
        }
        for (int n = 0; n < nsolve; ++n) {
            bp::CGResult res;
            if constexpr (!collocated) {
                if (transport) {
                    // A (u^{n+1} - u^n) = r_TG(u^n): the update starts from zero
                    A.advect(tg_a, tg_c, d_x, d_b);
                    bp::backend::fill(nL, T(0), d_delta);
                    res = bp::pcg<T>(A, cM, cK, pc, d_b, d_delta, w, o.rtol, o.maxit);
                    bp::backend::axpby(nL, T(1), d_delta, T(1), d_x);
                }
            }
            if (!transport) {
                if (o.problem == Problem::heat) {
                    // b = (M - (1 - theta) dt K) u^n, masked like the operator
                    A.apply(T(1), T(-(1.0 - o.theta) * o.dt), d_x, d_b);
                } else {
                    // b = M f (projection of the forcing / the initial function)
                    A.apply(T(1), T(0), d_f, d_b);
                }
                res = bp::pcg<T>(A, cM, cK, pc, d_b, d_x, w, o.rtol, o.maxit);
            }
            solve_seconds += res.seconds;
            total_iterations += res.iterations;
            ++solves;
            converged = converged && (res.converged || fixed_iterations);
            if (time_dependent) {
                std::printf("  step %3d: %4d iterations, ||r||/||b|| = %.2e\n",
                            n + 1, res.iterations, res.relative_residual);
                if (!o.vtk.empty()) {
                    bp::backend::from_device(nL, d_x);
                    vtk_ok = write_fields(step_file(o.vtk, n + 1), mesh, coords, x, exact, o.dt * (n + 1)) && vtk_ok;
                }
            } else {
                std::printf("  CG: %d iterations, ||r||/||b|| = %.2e%s\n",
                            res.iterations, res.relative_residual,
                            (res.converged || fixed_iterations) ? "" : " (not converged)");
            }
        }

        // errors against the manufactured solution: nodal maximum and the
        // M-norm sqrt(e^T M e), one more mass application
        bp::backend::from_device(nL, d_x);
        if (!time_dependent && !o.vtk.empty()) {
            vtk_ok = write_fields(o.vtk + ".vtk", mesh, coords, x, exact, 0.0);
        }
        for (std::size_t g = 0; g < nL; ++g) {
            r[g] = x[g] - u_exact[g];
        }
        err_max = max_abs(r);
        bp::backend::to_device(nL, d_r);
        const T* saved_mask = A.mask;
        A.mask = nullptr;
        A.apply(T(1), T(0), d_r, d_Ap);
        A.mask = saved_mask;
        err_M = std::sqrt(double(bp::backend::dot(nL, d_r, d_Ap)));
    }

    if (!vtk_ok) {
        return 1;
    }

    // ---- report ---------------------------------------------------------------
    std::printf("checks: 1^T M 1 = %.15g (volume 1), max |K 1| / max |K u| = %.2e\n",
                volume, k_const / k_scale);
    if (transport) {
        std::printf("        Taylor-Galerkin on u = e.x: max |r - a |e|^2 M 1| / max |a |e|^2 M 1| = %.2e\n",
                    tg_linear);
    }
    std::printf("error: max nodal |u - u_exact| = %.3e, ||u - u_exact||_M = %.3e\n",
                err_max, err_M);
    const double op_seconds = A.seconds();
    std::printf("solve: %ld CG iterations in %d solve%s, %.3f s, %.3f ms/iteration%s\n",
                total_iterations, solves, solves == 1 ? "" : "s", solve_seconds,
                1e3 * solve_seconds / std::max(1L, total_iterations),
                converged ? "" : " -- NOT CONVERGED");
    std::printf("  operator %ld applications %.3f s: gather %.3f, mass %.3f, stiffness %.3f, advect %.3f, combine %.3f, scatter %.3f, mask %.3f\n",
                A.applications, op_seconds, A.t_gather.seconds, A.t_mass.seconds,
                A.t_stiff.seconds, A.t_advect.seconds, A.t_combine.seconds, A.t_scatter.seconds, A.t_mask.seconds);
    std::printf("  vector ops %.3f s\n", std::max(0.0, solve_seconds - op_seconds));
    std::printf("throughput: %.3f MDoF/s (nodes x CG iterations / solve time)\n",
                1e-6 * double(nL) * double(total_iterations) / solve_seconds);
    return converged ? 0 : 2;
}

template <typename T>
int dispatch(const Options& o)
{
    // Runtime p -> compile-time nq, one instantiation per supported order
    // (BK1/BK3: nq = p + 2, BK5: nq = p + 1).
    if (o.collocated) {
        switch (o.p) {
            case 1: return run<T, 2, true>(o);
            case 2: return run<T, 3, true>(o);
            case 3: return run<T, 4, true>(o);
            case 4: return run<T, 5, true>(o);
            case 5: return run<T, 6, true>(o);
            case 6: return run<T, 7, true>(o);
            case 7: return run<T, 8, true>(o);
            default: break;
        }
    } else {
        switch (o.p) {
            case 1: return run<T, 3, false>(o);
            case 2: return run<T, 4, false>(o);
            case 3: return run<T, 5, false>(o);
            case 4: return run<T, 6, false>(o);
            case 5: return run<T, 7, false>(o);
            case 6: return run<T, 8, false>(o);
            case 7: return run<T, 9, false>(o);
            case 8: return run<T, 10, false>(o);
            default: break;
        }
    }
    std::cerr << "unsupported polynomial order p = " << o.p << "\n";
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 1;
    }
    return dispatch<real>(o);
}
