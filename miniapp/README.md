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
`mass` run must return its own input.

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
`--atomic` scatter with atomic adds, `--warp a` mesh deformation (|a| < 0.18),
`--tol`/`--maxit` CG control (`--tol 0` runs exactly `--maxit` iterations),
`--dt`/`--steps`/`--theta` for `heat`, `--vtk <base>` to write the fields
(below). `BP_PARALLEL` is not needed: without a
device the `target` regions run on the host threads (`OMP_NUM_THREADS`).

A run prints the problem size, the CG history, two sanity checks, the error
against the manufactured solution and a timing breakdown:

```
bp poisson: p = 3, nq = 5 (Gauss-Legendre, BK1/BK3), 8 x 8 x 8 elements, 15625 nodes, 32768 E-vector entries
precision 8 bytes, preconditioner none, scatter transpose map, warp 0.1
  CG: 91 iterations, ||r||/||b|| = 9.07e-09
checks: 1^T M 1 = 1 (volume 1), max |K 1| / max |K u| = 4.79e-16
error: max nodal |u - u_exact| = 8.342e-06, ||u - u_exact||_M = 1.554e-06
solve: 91 CG iterations in 1 solve, 0.321 s, 3.527 ms/iteration
  operator 97 applications 0.280 s: restrict 0.030, mass 0.004, stiffness 0.180, combine 0.000, prolong+mask 0.066
  vector ops 0.041 s
throughput: 4.43 MDoF/s (nodes x CG iterations / solve time)
```

- `1^T M 1` is the volume of the domain (1 for any warp) and `K 1` must vanish
  because constants have no gradient: both test the metric factors and the
  basis matrices independently of the solver.
- The errors decrease like h^(p+1) under refinement and exponentially in p;
  the `heat` error is dominated by the time step (first order for θ = 1,
  second for θ = 0.5).
- The throughput line is the CEED metric (degrees of freedom times
  iterations per second). On the uniform mesh the smooth manufactured
  solution is close to an eigenvector of the discrete Laplacian and CG stops
  after a few iterations; use `--warp`, larger meshes or `--tol 0 --maxit N`
  for timing.

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

## Structure

```
bk1_kernel.h, bk3_kernel.h, bk5_kernel.h   element kernels (shared with BK1/3/5.cpp)
miniapp/
  bp_basis.h     GLL/GL points and weights, Lagrange basis and derivative matrices
  bp_mesh.h      Cartesian mesh, node numbering, element restriction, Dirichlet mask
  bp_geometry.h  metric factors JxW and G in the kernels' layouts, element diagonals
  bp_backend.h   OpenMP target: restriction, prolongation, vector operations, dot
  bp_operator.h  A = cM M + cK K applied matrix-free, with phase timers
  bp_solver.h    preconditioned conjugate gradients
  bp_vtk.h       legacy VTK writer for the nodal fields
  bp.cpp         problems, options, checks and report
```

One operator application `y = A x` is

    E = P x                        restrict  (L-vector -> E-vector, a gather)
    E' = cM M_e E + cK K_e E       the kernels, element by element
    y = mask ∘ Pᵀ E'               prolong   (E-vector -> L-vector, a scatter-add)

The L-vector holds one value per global node, the E-vector one per element
and local node. `P` is stored as an index array (`e_to_l`), and `Pᵀ` as its
transpose in CSR form so each global node sums its own element entries: the
result is deterministic. `--atomic` uses the direct transpose with atomic
adds instead. Dirichlet conditions are imposed by masking: the boundary
rows of `A x` and of the right-hand side are zeroed, and all iterates keep
zero boundary values, so CG runs on the interior system.

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
`bp_backend.h`, which has eleven one-loop functions: `restrict_`,
`prolong`, `prolong_atomic`, `copy`, `fill`, `scale`, `axpby`, `lincomb`,
`pointwise`, `pointwise_inplace`, `dot`, plus `to_device`/`from_device`.
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

Exercises that fit a short tutorial: compare `--atomic` with the transpose
map at several orders; compare `BP_REAL=float` and double at fixed
iterations; fuse the mass and stiffness kernels for the `heat` operator so
the interpolation to quadrature points happens once (an "BK3 + BK1"
kernel); replace the Cartesian restriction by one read from a mesh file,
nothing else needs to change.

## Why a hand-written CG

The solver is 60 lines (`bp_solver.h`) and needs only the operator and six
vector operations, which is exactly the set a port must provide, so the
same file runs unchanged on every backend. Library solvers that can work
matrix-free exist (PETSc's KSP with a shell matrix, Ginkgo, the
reverse-communication CG of classic Intel MKL on the CPU) but each pulls in
its own vector type and device model; for a tutorial about the kernels that
is more to explain than it saves.
