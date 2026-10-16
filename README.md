# omp_bk

OpenMP `target`-offload implementations of the CEED "bake-off" kernels
(sum-factorized spectral-element operators):

| Program | Operator                                   |
|---------|--------------------------------------------|
| `BK1`   | Mass matrix                                |
| `BK3`   | Poisson (stiffness) matrix                 |
| `BK5`   | Collocated Laplacian at quadrature points  |
| `BK1_ff`| Mass matrix in float-float arithmetic (see below) |

## Requirements

- A C++17 compiler with OpenMP support (e.g. GCC or Clang).
- To run the kernels on a GPU, an OpenMP-offload-capable toolchain
  (e.g. GCC built with `nvptx`/`amdgcn` offload, or Clang with the matching
  offload runtime). Without a GPU the `target` regions run as a host fallback.

## Build

```sh
make            # builds BK1, BK3, BK5, BK1_ff
make clean      # removes the executables
```

The default flags live in `Makefile` (`CXX`, `CXXFLAGS`). Override them from
the command line, e.g. to use Clang:

```sh
make CXX=clang++
```

## Run

```
./BK1 <p> [nelmt] [ntests]      # BK1/BK3: p is the polynomial order, nq = p + 2
./BK3 <p> [nelmt] [ntests]
./BK5 <nq> [nelmt] [ntests]     # BK5: first argument is nq directly
```

- `nelmt`  — number of elements (default 524288).
- `ntests` — timing repetitions; the minimum wall time is reported (default 5).

Each run prints the achieved `GDoF/s` and effective `GB/s`, followed by the
solution norm (useful as a quick correctness check).

## Float-float version of BK1

`BK1_ff` runs the BK1 sum factorization in float-float arithmetic. The
kernel `SumFactorizationFF` in `BK1_ff.cpp` has the interface of
`SumFactorization` in double: the caller passes IEEE doubles for the basis,
the weights and the input and gets doubles back. Inside the kernel every
value is split into a pair of floats (hi + lo, ~48 significant bits, float's
exponent range) as it is loaded, the nine steps run on separate hi and lo
work arrays, and the result is joined back into a double as it is stored.
Every multiply and multiply-add is a short chain of fp32 adds, multiplies
and fused multiply-adds built on the error-free transformations TwoSum and
TwoProd; the helpers live in `float_float.h`, a private implementation
header of the driver with no pair type in any interface. A contraction
accumulates with `mul_acc` (12 flops: the product of Algorithm 11 of
Joldes, Muller and Popescu 2017 without its renormalisation, added with
Dekker's add2 without its renormalisation) and renormalises the running
sum once per column (`renormalize`, 3 flops), so the lo part of the sum
grows by at most about one ulp per term before it is folded back. On GPUs
whose fp64 rate is 1/32 or 1/64 of fp32, this recovers near-double accuracy
at about 12 fp32 flops per multiply-add while moving the same 8 bytes per
value as double (the split and join cost one fp64 operation per value
loaded or stored).

There are two float-float kernels with the same arithmetic. `float-float`
runs one element per thread with the work arrays in thread-local memory, as
`BK1` does. `ff-teams` (`SumFactorizationFFTeams`) runs one element per
team: the hi and lo work arrays are declared at team scope, so they are
shared by the team (shared memory on a GPU), the points of each step are
spread over the team's threads with `parallel for collapse(3)`, and the end
of each parallel region is the barrier between steps. The team size is a
constexpr of nq (`ff_team_size`: nq^3 rounded up to whole warps, 1024 at
nq = 10). `ff-plane` (`SumFactorizationFFPlane`) is the layout of the
production codes: a plane of nq x nq threads per element (the team rounded
up to whole warps), each thread owning the column through its point in
registers (hi and lo of nq values),
five phases with a barrier between them, the contraction index as the outer
loop so that every input value and basis entry is loaded once per column,
and the basis read as a warp broadcast. The plane kernel runs twice, with
one element per team and with a batch of elements per team (`ff-plane xN`),
which fills the team at low order; the default batch fills about 128
threads (`ff_plane_batch`: 14 at nq = 3, 8 at nq = 4, 1 from nq = 9) and
`make BK1_ff CPPFLAGS=-DFF_PLANE_BATCH=n` forces one batch for every order.
`ff-loop` (`SumFactorizationFFPlaneLoop`) is the same plane kernel written
with `teams loop` and a nested `loop` per phase instead of a parallel
region and worksharing loops: the descriptive form that nvc++ lowers like
OpenACC gang and vector loops, with the barriers between phases placed by
the compiler. It shares the phase bodies with `ff-plane` and takes the same
batch. All float-float kernels sum every output in the same index order and
give the same result bit for bit. On the host the team kernels are slow,
`ff-teams` and `ff-plane` open parallel regions per team and `ff-loop`
runs a team on one thread; they are meant for the GPU.

The driver runs double, float and the float-float kernels on the same
data, prints one row per kernel and the relative L2 error of float (~1e-7)
and of each float-float kernel (~3e-15) against the double result, all
reduced in double. The last line is the check of the float-float results,
`ok` or `FAIL` against a tolerance (default 1e-10, `BK_TOL=...`), with exit
status 1 on failure:

```sh
./BK1_ff <p> [nelmt] [ntests]     # same arguments and constant data as BK1; the float row prints BK1's norm
BK_RANDOM=1 ./BK1_ff 2            # pseudo-random data (the same as BK1_amx's)
BK_CSV=rows.csv ./BK1_ff 2        # also append one CSV row per kernel to rows.csv (all drivers)
for p in 1 2 3 4 5 6 7 8; do ./BK1_ff $p 4096 1 | tail -1; done   # the check per order
```

The error-free transformations need strict IEEE semantics: the header
refuses `-ffast-math`, anything subtler shows up as `FAIL`, and the Makefile
selects the native instruction set so `std::fma` is an instruction (the
first output line says `fma = hardware`).

### CUDA kernel

`bk1_ff_cuda.cuh` holds the same float-float kernel as a CUDA kernel,
`MassOperatorFF<nq>`, with double storage at the boundary: one block works
through batches of `nelmtPerBatch` elements, a thread owns one point of an
element's plane with the column through it in registers, the five phases
are those of the OpenMP plane kernel, and the boxes between the phases are
hi/lo float pairs in dynamic shared memory with the one nq^3 layout of the
OpenMP kernels. `mass_operator_ff_traits<nq>` holds the sizes per element
and the launch geometry: threads per element, block size, dynamic shared
memory bytes (the same count as a double kernel's) and the default batch.
Boxes and global arrays are addressed through accessor lambdas. The loops
over a register column are fully unrolled; the loops over the contraction
index are unrolled fully up to nq = 5 and at nq = 9 and by a factor of 2
elsewhere (`mass_operator_ff_traits<nq>::contraction_unroll`, the better
setting measured per order on GH200; `-DFF_CUDA_CONTRACTION_UNROLL=n` sets
one factor for every order, 0 for full). The header is empty unless
compiled by nvcc, and it needs no special flags: the error-free transformations survive nvcc's
default contraction, and `--use_fast_math` only flushes subnormal lo parts.
The header comment lists the launch geometry and the known cost.
Two drivers run the kernel on the GPU, take the arguments of `BK1_ff`
plus the elements per block, time the kernel with CUDA events after warm-up
runs worth 0.2 s of kernel time (`BK_WARMUP=s` changes it; the GPU drops
to its idle clock while the host computes the reference, and the minimum
of a few short runs then depends on how far the clock has ramped) and
check it against the host double reference (`ok`/`FAIL`); they share
`cuda/bk1_ff_driver_common.h`. `make BK1_ff_cuda_plain` builds
`cuda/BK1_ff_cuda_plain.cu` on the classic CUDA runtime API, for any nvcc
(`NVCC=...` selects one, `NVARCH=native` is the default architecture).
`make BK1_ff_cuda` builds `cuda/BK1_ff_cuda.cu` on the CCCL runtime API of
CUDA 13.2 or later (`CCCL=/path/to/cccl` puts a newer CCCL checkout ahead
of the toolkit's): the kernel is a functor whose call operator takes the
launch configuration as a template parameter and reads the block and
thread ranks and the dynamic shared memory from it. With glibc 2.41 or
later the headers of the toolkits up to 12.8 fail in nvcc on the
`noexcept` of `cospi`, `sinpi` and `rsqrt`;
`cuda/cuda_headers_glibc241.sh <toolkit include dir> <destination>` makes
a patched copy of the include tree to pass as `NVCCEXTRA="-I<destination>"`
(the 11.8 headers do not have the problem).
`MassOperatorFF2` in the same header is the experiment on the pair layout:
the same kernel, operation for operation, with every float-float value as
one `float2` (x = hi, y = lo) in shared memory and in registers, and the
helpers of `float_float2.h` taking and returning pairs by value through
explicit round-to-nearest intrinsics, in the style of NVIDIA's `dbldbl.h`;
a box value is then one 8-byte shared-memory access instead of two 4-byte
ones. The plain driver runs both (`float-float` and `ff-float2` rows) and
checks that their outputs are identical.
`make check_ff_cuda && ./check_ff_cuda` builds and runs both kernels on
the host through `cuda/cuda_shim.h`, a stand-in for the CUDA keywords with one
thread per block, and checks them against the double reference for
nq = 3..10, constant and random data and batches with a short last batch,
and against each other bit for bit (`ok`/`FAIL`, exit status 1 on
failure); they agree with the OpenMP kernels bit for bit. The stand-in
checks indexing, phases and batching, not the execution by many threads.

### Measurement scripts

All drivers append one CSV row per kernel and run to the file named by
`BK_CSV` (`driver,kernel,nq,nelmt,batch,gdofs,gbs,norm,error,ok`, header
first when the file is new), so a sweep is a shell loop and nothing has to
be parsed. The scripts in `cuda/scripts/` are built on that and on Nsight
Compute's report files; none of them interprets the drivers' text output.

```sh
cuda/scripts/sweep.sh sweep.csv ./BK1_ff_cuda_plain ./BK1_ff   # ORDERS="2 3 4 5 6 7 8" REPEAT=1 NELMT=1000000 NTESTS=5
cuda/scripts/table.py sweep.csv                 # best GDoF/s by kernel and nq, the run-to-run spread, failed checks
cuda/scripts/table.py sweep.csv error           # the relative errors instead
make -B BK1_ff_cuda_plain NVCCEXTRA='-Xptxas -v' 2>&1 | cuda/scripts/regs.sh   # registers, stack, spills per kernel
nvc++ ... -gpu=ptxinfo -c BK1_ff.cpp 2>&1 | cuda/scripts/regs.sh              # the same for the OpenMP kernels
cuda/scripts/ncu_stats.py --orders 7 8          # Nsight Compute statistics of both CUDA kernels at nq = 9, 10
cuda/scripts/ncu_stats.py --binary ./BK1_ff --kernels regex:PlaneLoopILi10ELi1E --skip 0 --orders 8
cuda/scripts/sass_count.sh BK1_ff_cuda_plain BK1_ff   # instructions per kernel in the SASS
cuda/scripts/run_experiments.sh                 # all of the above into results/<date>/
BK1_FF=./BK1_ff NVCXX="nvc++ <flags>" REPEAT=2 cuda/scripts/run_experiments.sh
```

`sweep.sh` runs each driver for every order (and `REPEAT` times) with
`BK_CSV` set and ends with the table; `table.py` pivots the CSV with the
csv module. `regs.sh` is an awk over the fixed strings of a ptxas report,
with `cu++filt` for the names when it is on the path. `ncu_stats.py`
has ncu write a report per kernel and order with a fixed metric list
(duration, throughputs, issue-slot utilisation, eligible and active warps
per scheduler, achieved occupancy, registers and occupancy limits,
executed instructions, pipe utilisation of fma, fp64, alu and lsu,
shared-memory loads, stores and bank conflicts, warp stall reasons),
skipping the warm-up launches, reads the reports through Nsight Compute's
Python Report Interface (`ncu_report`, found in `extras/python` next to
the `ncu` binary, or given with `--python-path`) and shows one table per
nq with a column per kernel; the reports stay in `ncu_reports/` for
`ncu-ui`, `--load` tabulates existing reports side by side, metrics the
local `ncu --query-metrics` does not list are left out, `--drop` leaves
out more, and the OpenMP kernels are selected with a `regex:` name. `run_experiments.sh` chains them into a
results directory with the build log, the drivers' output and the GPU and
driver versions; its environment variables select orders, repeats,
element counts, extra nvcc flags, the OpenMP driver and an nvc++ command
for the OpenMP register table and unroll messages. `results/` is ignored
by git.

## Apple AMX version of BK1

`amx/BK1_amx.cpp` runs the BK1 sum factorization on the Apple AMX
coprocessor; elsewhere `amx/amx.h` emulates the AMX instructions in
software, so the kernel logic can be checked on any machine. The same binary
also runs the CPU baselines (`BK_KERNEL=neon`: 4-wide NEON FMAs across the
elements of a batch; `BK_KERNEL=refv`: the reference with unit-stride inner
loops). The header of `amx/BK1_amx.cpp` lists all environment variables.

```sh
make BK1 BK1_amx bw_test amx_pipe     # accel_gemm needs macOS (Accelerate)
./BK1_amx <p> [nelmt] [ntests]        # same arguments as BK1, p = 1..14
BK_KERNEL=neon BK_LAYOUT=soa ./BK1_amx 4

amx/scripts/validate.sh               # every kernel against the reference; exit 1 on failure
amx/scripts/throughput.sh results.csv # sweep (all kernels from BK1_amx): kernel,p,target,nelmt,dofs,gdofs,gbs
amx/scripts/high_order.sh results_high.csv   # the same for p = 9..14 (AMX binary only)
OUT=tune.csv amx/scripts/tune.sh      # best layout/path/batch per order
./bw_test    > bw.csv                 # bandwidth and FMA roofs: test,threads,bytes,seconds,GB_per_s
./amx_pipe   > pipe.txt               # AMX load-to-use and store behaviour (one unit)
./accel_gemm > gemm.csv               # Accelerate sgemm calibration (macOS)

amx/scripts/plot_throughput.py results.csv throughput.png   # per kernel and order, AMX/NEON speedup
amx/scripts/plot_roofline.py results.csv bw.csv roofline.png # plateaus against the roofs
```

The scripts find `BK1` and `BK1_amx` in the repository root and run from
any directory; set `BK1_AMX=...` or `BK1=...` to use other builds. The
plots need matplotlib.
