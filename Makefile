CXX=g++
# Native ISA: -march=native on x86 (where -mcpu= is only a tuning alias and
# would leave std::fma a library call), -mcpu=native on AArch64 (Apple clang
# does not accept -march=native).
ifeq ($(shell uname -m),x86_64)
ARCHFLAGS=-march=native
else
ARCHFLAGS=-mcpu=native
endif
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 $(ARCHFLAGS) -fopenmp
# The CUDA driver (make BK1_ff_cuda): nvcc from CUDA 13.2 or later for the
# cuda::launch runtime API; NVARCH selects the GPU (native = the one present).
NVCC=nvcc
NVARCH=native
# With NVCC=nvc++ the CUDA sources are compiled by nvc++ -cuda (no host
# compiler version check; NVARCH is then a -gpu target such as cc90, or
# native for the GPU present at build time; NVCCEXTRA='-gpu=cuda11.8'
# selects the bundled 11.8 toolkit, whose headers have no glibc 2.41 clash).
# NVCCPTXINFO is the flag for the ptxas resource report of either compiler.
ifneq (,$(findstring nvc++,$(NVCC)))
NVCCFLAGS=-O3 -std=c++17 -cuda $(if $(filter native,$(NVARCH)),,-gpu=$(NVARCH))
NVCCPTXINFO=-gpu=ptxinfo
else
NVCCFLAGS=-O3 -std=c++17 -arch=$(NVARCH) -Xcompiler -Wall,-Wno-unknown-pragmas
NVCCPTXINFO=-Xptxas -v
endif
# CCCL=/path/to/cccl puts a checkout of CCCL (3.2 or later) ahead of the
# toolkit's own copy when that is too old for the runtime API.
CCCL=
NVCCFLAGS+=$(if $(CCCL),-I$(CCCL)/libcudacxx/include -I$(CCCL)/cub -I$(CCCL)/thrust,)
# NVCCHOST=/path/to/g++ selects nvcc's host compiler (-ccbin): nvcc 11.8
# accepts GCC up to 11, 12.x up to 12 or 13, and the first g++ on the PATH
# may be newer.
NVCCHOST=
NVCCFLAGS+=$(if $(NVCCHOST),-ccbin $(NVCCHOST),)
# NVCCEXTRA adds flags, e.g. NVCCEXTRA='-Xptxas -v -DFF_CUDA_CONTRACTION_UNROLL=0'
NVCCEXTRA=
NVCCFLAGS+=$(NVCCEXTRA)


.PHONY: all
all: BK1 BK3 BK5 BK1_ff


BK1: BK1.cpp bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# BK1 in float-float arithmetic (the kernel in BK1_ff.cpp, the helpers in
# float_float.h) next to double and float; never with -ffast-math (the
# header rejects it).
# CPPFLAGS=-DFF_PLANE_BATCH=n sets the elements per team of the plane kernel.
BK1_ff: BK1_ff.cpp float_float.h bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

BK3: BK3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# The CUDA float-float kernel (bk1_ff_cuda.cuh) run on the host through a
# stand-in for the CUDA keywords and checked against the double reference;
# exit status 1 on failure.  The real kernel needs nvcc and a GPU.
check_ff_cuda: cuda/check_ff_cuda.cpp cuda/cuda_shim.h bk1_ff_cuda.cuh float_float.h float_float2.h bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -Wno-unknown-pragmas -D__CUDACC__ -I. -Icuda -o $@ $<

# The CUDA float-float kernel on the GPU, timed and checked against the
# double reference on the host; needs nvcc (CUDA 13.2+) and a GPU.
BK1_ff_cuda: cuda/BK1_ff_cuda.cu cuda/bk1_ff_driver_common.h bk1_ff_cuda.cuh float_float.h float_float2.h bk1_sumfact.h bk_common.h
	$(NVCC) $(NVCCFLAGS) -I. -Icuda -o $@ $<

# The same on the classic CUDA runtime API, for any nvcc (11.8 included).
BK1_ff_cuda_plain: cuda/BK1_ff_cuda_plain.cu cuda/bk1_ff_driver_common.h bk1_ff_cuda.cuh float_float.h float_float2.h bk1_sumfact.h bk_common.h
	$(NVCC) $(NVCCFLAGS) -I. -Icuda -o $@ $<

# Apple AMX version of BK1 (software-emulated AMX on any other machine)
BK1_amx: amx/BK1_amx.cpp amx/amx.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Micro-benchmarks for the roofline: memory bandwidth and FMA peaks, the AMX
# unit's load/store pipeline, and (macOS only) Accelerate's sgemm.
bw_test: amx/bench/bw_test.cpp amx/amx.h
	$(CXX) $(CXXFLAGS) -o $@ $<

amx_pipe: amx/bench/amx_pipe.cpp amx/amx.h
	$(CXX) $(CXXFLAGS) -o $@ $<

accel_gemm: amx/bench/accel_gemm.cpp
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -o $@ $< -framework Accelerate

.PHONY: clean
clean:
	$(RM) BK1 BK3 BK5 BK1_ff BK1_amx check_ff_cuda BK1_ff_cuda BK1_ff_cuda_plain bw_test amx_pipe accel_gemm *.o
