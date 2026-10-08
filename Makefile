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


.PHONY: all
all: BK1 BK3 BK5 BK1_ff


BK1: BK1.cpp bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# BK1 in float-float arithmetic (float_float.h) next to double and float;
# never with -ffast-math (the header rejects it).
BK1_ff: BK1_ff.cpp float_float.h bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

BK3: BK3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

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
	$(RM) BK1 BK3 BK5 BK1_ff BK1_amx bw_test amx_pipe accel_gemm *.o
