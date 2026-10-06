# bp: solving the bake-off problems with the BK kernels

`bp` is a small finite-element solver built on the element kernels of this
repository. It turns the "bake-off kernels" (BK1, BK3, BK5: the element-local
mass and stiffness operators) into the "bake-off problems" of the CEED
project (BP1, BP3, BP5: linear systems solved with conjugate gradients) and
adds an implicit diffusion problem that needs the mass and the stiffness
operator in every iteration.

| Problem   | System solved per CG run                              | Kernels              |
|-----------|-------------------------------------------------------|----------------------|
| `mass`    | M u = M f, so u = f (BP1)                             | BK1                  |
| `poisson` | K u = M f, homogeneous Dirichlet (BP3)                | BK3, BK1 for the rhs |
| `poisson --gll` | the same, collocated on GLL points (BP5)        | BK5, lumped mass     |
| `heat`    | (M + θ Δt K) uⁿ⁺¹ = (M − (1 − θ) Δt K) uⁿ, one solve per step | BK1 + BK3 (or lumped mass + BK5) |
| `transport` | M (uⁿ⁺¹ − uⁿ) = r_TG(uⁿ), Taylor-Galerkin for u_t + e·∇u = 0; `--tg3`: M + Δt²/6 K_e | tg_kernel.h + BK1 (+ BK3); `--gll`: lumped mass (+ BK5) |

The mesh is a Cartesian partition of the unit cube into hexahedra, optionally
deformed by a smooth map (`--warp`) so that the metric factors vary within
each element. Degrees of freedom sit on the Gauss-Lobatto-Legendre (GLL)
nodes of each element; the Gauss-Legendre family integrates with q = p + 2
points per direction (BK1/BK3), the collocated family uses the q = p + 1 GLL
nodes themselves (BK5). The manufactured solution is

    u(x, t) = exp(-3 π² t) sin(π x) sin(π y) sin(π z)

which solves −Δu = 3π² u at t = 0 and u_t = Δu with f = 0, so the `poisson`
run checks the operators against an exact answer, the `heat` run also checks
the time integration (θ = 1 backward Euler, θ = 0.5 Crank-Nicolson), and the
`mass` run must return its own input. The `transport` run moves a Gaussian
bump exp(−|x − x₀ − e t|² / 2σ²) with a constant velocity e, placed so that
it passes the centre of the cube half-way through the run and stays away
from the walls.

## Build and run

```sh
make bp                       # same flags as the kernels; -DBP_REAL=float for single precision
./bp poisson -p 4 -n 16       # BP3: order 4, 16^3 elements
./bp poisson -p 4 -n 16 --gll --pc jacobi
./bp heat -p 3 -n 8 --dt 1e-3 --steps 10 --theta 0.5
./bp mass -p 3 -n 8
./bp poisson -p 6 -n 8 --warp 0.1 --tol 0 --maxit 100   # fixed 100 iterations: benchmark mode
./bp                           # option summary on a bad argument
```

Options: `-p` order (1..8, 1..7 with `--gll`), `-n <n>` or `-n <nx,ny,nz>`
elements, `--gll` collocated family, `--pc jacobi` diagonal preconditioner,
`--atomic` scatter with atomic adds, `--fused` gather on the fly (below),
`--warp a` mesh deformation (|a| < 0.18),
`--tol`/`--maxit` CG control (`--tol 0` runs exactly `--maxit` iterations),
`--dt`/`--steps`/`--theta` for `heat`, `--velocity ex,ey,ez`/`--sigma`/`--tg3`
for `transport`, `--vtk <base>` to write the fields (below). `BP_PARALLEL` is not needed: without a
device the `target` regions run on the host threads (`OMP_NUM_THREADS`).

A run prints the problem size, the CG history, two sanity checks, the error
against the manufactured solution and a timing breakdown:

```
bp poisson: p = 3, nq = 5 (Gauss-Legendre, BK1/BK3), 8 x 8 x 8 elements, 15625 nodes, 32768 E-vector entries
precision 8 bytes, preconditioner none, scatter transpose map, warp 0.1
  CG: 91 iterations, ||r||/||b|| = 9.07e-09
checks: 1^T M 1 = 1 (volume 1), max |K 1| / max |K u| = 4.79e-16
error: max nodal |u - u_exact| = 8.342e-06, ||u - u_exact||_M = 1.554e-06
solve: 91 CG iterations in 1 solve, 0.238 s, 2.621 ms/iteration
  operator 0.160 s inside the solves, vector ops 0.078 s
  operator 96 applications in total 0.166 s: gather 0.024, mass 0.002, stiffness 0.086, advect 0.000, scatter 0.029, mask 0.026
throughput: 5.96 MDoF/s (nodes x CG iterations / solve time)
```

- `1^T M 1` is the volume of the domain (1 for any warp) and `K 1` must vanish
  because constants have no gradient: both test the metric factors and the
  basis matrices independently of the solver.
- The errors decrease like h^(p+1) under refinement and exponentially in p;
  the `heat` error is dominated by the time step (first order for θ = 1,
  second for θ = 0.5).
- The first operator line counts only the applications inside the CG
  solves, so that line and the vector operations add up to the solve time;
  the second is the phase breakdown over every application, including the
  checks and the right-hand sides.
- The throughput line is the CEED metric (degrees of freedom times
  iterations per second). On the uniform mesh the smooth manufactured
  solution is close to an eigenvector of the discrete Laplacian and CG stops
  after a few iterations; use `--warp`, larger meshes or `--tol 0 --maxit N`
  for timing.

## The Taylor-Galerkin transport step

`tg_kernel.h` is a BK-style kernel for the explicit Taylor-Galerkin
(Lax-Wendroff) step of u_t + e·∇u = 0 with a constant velocity, as in
lattice-Boltzmann streaming:

    r = ∫ φ a (e·∇u) + ∫ (e·∇φ) c (e·∇u),   a = −Δt,  c = −Δt²/2,
    M (uⁿ⁺¹ − uⁿ) = r              (`--tg3`: (M + Δt²/6 K_e)(uⁿ⁺¹ − uⁿ) = r)

The kernel has the loop structure of BK3 (interpolate, differentiate,
point operation, integrate values and gradients, project back) but reads
no metric data: the velocity enters as its reference image ẽ = J⁻¹e,
constant on the Cartesian mesh, and e·∇u is the combination of the three
reference derivatives weighted by ẽ. A zero component skips that
contraction, so an axis direction costs one derivative pass instead of
three. The point operation is the two lines that scale e·∇u by a and by c;
everything else is the generic sum factorisation. Like BK3 the kernel is
templated on the node count: with `--gll` the nodes are the quadrature
points, the interpolation and projection steps are compiled out, and the
mass matrix is the lumped (diagonal) one, so the TG2 step is one kernel
application and a pointwise divide with no CG at all; `--tg3` still needs
CG for M + Δt²/6 K_e. The lumped mass is cheaper and less accurate: at
p = 6 on 8³ elements with Δt = 2.5e-3 the error after 160 steps is 1.4e-4
with the consistent mass and TG3, 5.2e-4 lumped with TG3, 2.8e-3 lumped
with TG2.

The `--tg3` solve reuses BK3 unchanged: the directional stiffness
K_e = ∫ (e·∇φ)(e·∇u) is BK3 with the rank-one metric G = |J| w ẽẽᵀ, which
`make_geometry` produces when given the direction. Each run checks the
kernel on the linear function u = e·x, for which r equals a|e|² M 1 on the
interior nodes exactly (the flux term integrates to a wall contribution
only); the reported ratio must be at round-off. The wall surface term of
the integration by parts is not implemented and the walls carry u = 0, so
the bump must stay away from them: its tail at the inflow wall is the one
inconsistency with the exact solution, 4e-6 for the default σ = 0.06 and
already 9e-4 for σ = 0.08, which then bounds the error of any resolution.
The error of a converged run trails the bump as the dispersive ripple of
the scheme and peaks each time the bump crosses an element boundary.

## Plotting the fields

`--vtk <base>` writes the solution `u`, the exact solution `u_exact` and
their difference `error` at the GLL nodes as a legacy ASCII VTK file
(`<base>.vtk`; the `heat` problem writes `<base>_0000.vtk` for the initial
condition and one file per step). The file is a `STRUCTURED_GRID` of the
nnode_x × nnode_y × nnode_z nodes with explicit coordinates, so warped
meshes plot correctly; ParaView and VisIt open it directly, and the
per-step files of a `heat` run form a time series.

```sh
./bp heat -p 3 -n 8 --steps 20 --theta 0.5 --vtk out/heat   # out/ must exist
./bp poisson -p 4 -n 8 --warp 0.1 --vtk poisson
```

`miniapp/plot_vtk.py` (numpy and matplotlib) draws these files without
ParaView: for one file the mid-plane slices of `u` and of the error on the
node coordinates plus the centre-line profile against the exact solution,
for the files of a `heat` run the profiles over time and the maximum error
per step.

```sh
miniapp/plot_vtk.py poisson.vtk poisson.png
miniapp/plot_vtk.py out/heat_00*.vtk heat_series.png
```

## Structure

```
bk1_kernel.h, bk3_kernel.h, bk5_kernel.h   element kernels (shared with BK1/3/5.cpp)
tg_kernel.h                                Taylor-Galerkin advection kernel (bp only)
miniapp/
  bp_basis.h     GLL/GL points and weights, Lagrange basis and derivative matrices
  bp_mesh.h      Cartesian mesh, node numbering, element restriction, Dirichlet mask
  bp_geometry.h  metric factors JxW and G in the kernels' layouts, element diagonals
  bp_backend.h   OpenMP target: gather, scatter-add, vector operations, dot
  bp_timer.h     accumulating stopwatch for the phase breakdown
  bp_operator.h  A = cM M + cK K applied matrix-free, with phase timers
  bp_solver.h    preconditioned conjugate gradients
  bp_vtk.h       legacy VTK writer for the nodal fields
  plot_vtk.py    matplotlib plots of those files (slices, profiles, time series)
  bp.cpp         problems, options, checks and report
```

One operator application `y = A x` is

    E = P x                        gather       (L-vector -> E-vector)
    E' = cM M_e E + cK K_e E       the kernels, element by element
    y = mask ∘ Pᵀ E'               scatter-add  (E-vector -> L-vector)

The L-vector holds one value per global node, the E-vector one per element
and local node. `P` is stored as an index array (`e_to_l`), and `Pᵀ` as its
transpose in CSR form so each global node sums its own element entries: the
result is deterministic. `--atomic` uses the direct transpose with atomic
adds instead. Each kernel's output is scattered with its coefficient, so
c_M M + c_K K is two kernel passes on one gathered E-vector and two
accumulating scatters. The collocated (GLL) mass matrix is diagonal, so it
is assembled once as an L-vector and applied without any restriction. (`P` is libCEED's element restriction; it only changes the
storage layout and has nothing to do with multigrid.) Dirichlet conditions
are imposed by masking: the boundary rows of `A x` and of the right-hand
side are zeroed, and all iterates keep zero boundary values, so CG runs on
the interior system.

### E-vector or gather on the fly

Each kernel header has two entry points with the same loop nest; the
element loop and the contraction loops stay in one kernel body so a
compiler can parallelise across elements, and the two bodies differ only
in their first and last step:

- `SumFactorization`, the CEED bake-off structure: the E-vector is a stored
  array, the kernel reads its slice and writes its slice, and the gather
  and scatter are separate passes. The kernel's in/out traffic goes
  through global memory twice.
- `SumFactorizationFused` (`--fused`): the first step gathers the
  element's input box from the L-vector through `e_to_l` and the last
  step adds alpha times the output box into the L-vector with atomic
  updates. Nothing element-sized is stored; the cost is the atomics and a
  summation order that varies between runs (the CG iteration count may
  change by one). The combination `cM M + cK K` comes for free: each
  kernel adds its own multiple into the same output.

The timing breakdown separates the two: compare `gather + scatter` of the
E-vector path against the extra time inside `mass` and `stiffness` of the
fused path. On a 4-core CPU host fallback at p = 4, 12³ elements, `heat`
for 60 iterations:

```
E-vector, transpose map    9.7 ms/iter   gather 0.030 mass 0.122 stiffness 0.283 scatter 0.072
E-vector, atomic scatter  10.3 ms/iter   gather 0.026 mass 0.112 stiffness 0.259 scatter 0.149
fused                      9.4 ms/iter   gather 0     mass 0.148 stiffness 0.309 scatter 0.016 (zeroing y)
```

On the CPU the atomics cost about what the stored E-vector saves; on a GPU
the balance shifts with the bandwidth and the atomic throughput of the
device, which is the point of measuring it.

The Jacobi diagonal is assembled once at setup: the diagonal of a
tensor-product element matrix `Bᵀ W B` is `(B ∘ B)ᵀ w`, and the six metric
terms of the stiffness pair `B` with its derivative `DB` in the same way
(`contract3` in `bp_geometry.h`). `Pᵀ` of the element diagonals is the
diagonal of the assembled operator.

Everything in `bp_basis.h`, `bp_mesh.h` and `bp_geometry.h` runs once on the
host in double precision. The solve runs entirely inside one `target data`
region: the only transfers are the solution at the end and the vectors of
the error checks.

## Porting to another programming model

The device code is confined to the three kernel headers and
`bp_backend.h`, which has twelve one-loop functions: `gather`,
`scatter_add`, `scatter_add_atomic`, `copy`, `fill`, `axpby`, `lincomb`,
`pointwise`, `pointwise_add`, `pointwise_inplace`, `dot` and the fused
conjugate-gradient update `cg_update`, plus `to_device`/`from_device`.
An OpenACC, Kokkos or CUDA version replaces those files and the `target
data` region in `bp.cpp` (device allocation and the initial copies) and
keeps the rest. Things to carry over:

- The kernels assume one element per team/thread block/work item with the
  element's scratch arrays in registers or local memory; `Pᵀ` by transpose
  map is a plain gather, `Pᵀ` by atomics needs `atomicAdd` on the vector type.
- The dot product reduces in the working precision on the device; in
  single precision long sums lose accuracy, which is why float builds use a
  looser tolerance (`--tol 1e-5`).
- All backend functions are synchronous so the phase timers in
  `bp_operator.h` are meaningful; with asynchronous launches, fence before
  stopping a timer.

Exercises that fit a short tutorial: compare `--atomic`, the transpose
map and `--fused` at several orders; compare `BP_REAL=float` and double at
fixed iterations; fuse the mass and stiffness kernels for the `heat`
operator so the interpolation to quadrature points happens once (a
"BK3 + BK1" kernel); compute the global index arithmetically in the fused
path instead of reading `e_to_l`; replace the Cartesian restriction by one
read from a mesh file, nothing else needs to change.

## Why a hand-written CG

The solver is 60 lines (`bp_solver.h`) and needs only the operator and six
vector operations, which is exactly the set a port must provide, so the
same file runs unchanged on every backend. Library solvers that can work
matrix-free exist (PETSc's KSP with a shell matrix, Ginkgo, the
reverse-communication CG of classic Intel MKL on the CPU) but each pulls in
its own vector type and device model; for a tutorial about the kernels that
is more to explain than it saves.
