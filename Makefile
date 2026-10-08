CXX=g++
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 -mcpu=native -fopenmp


.PHONY: all
all: BK1 BK3 BK5 BK1_ff


BK1: BK1.cpp bk1_sumfact.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# BK1 in float-float arithmetic (float_float.h), timed against double and
# float.  Needs strict IEEE semantics (no -ffast-math); on x86 add -mfma or
# -march=native to CXXFLAGS so std::fma is an instruction.
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
