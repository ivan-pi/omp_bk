CXX=g++
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 -mcpu=native -fopenmp

# The arithmetic-intensity tool counts flops on the host with an
# operation-counting scalar type; it needs no OpenMP offload, and the
# `#pragma omp` lines in the kernels are intentionally ignored here.
AI_CXXFLAGS=-Wall -pedantic -O2 -std=c++17 -Wno-unknown-pragmas


.PHONY: all
all: BK1 BK3 BK5


BK1: BK1.cpp bk1_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

BK3: BK3.cpp bk3_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp bk5_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Exact flop / byte / arithmetic-intensity report (host-only, no offload).
arith_intensity: arith_intensity.cpp flop_counter.h \
                 bk1_kernel.h bk3_kernel.h bk5_kernel.h bk_common.h
	$(CXX) $(AI_CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) BK1 BK3 BK5 arith_intensity *.o
