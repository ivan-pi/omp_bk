CXX=g++
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 -mcpu=native -fopenmp


.PHONY: all
all: BK1 BK3 BK5


BK1: BK1.cpp bk1_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

BK3: BK3.cpp bk3_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp bk5_kernel.h bk_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Mini-app: CG solves of the bake-off problems (mass, Poisson, diffusion)
# with the same kernels.  BP_REAL=float selects single precision.
BP_HEADERS = miniapp/bp_basis.h miniapp/bp_mesh.h miniapp/bp_geometry.h miniapp/bp_backend.h \
             miniapp/bp_operator.h miniapp/bp_solver.h miniapp/bp_vtk.h
bp: miniapp/bp.cpp $(BP_HEADERS) bk1_kernel.h bk3_kernel.h bk5_kernel.h tg_kernel.h bk_common.h
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
	$(RM) BK1 BK3 BK5 bp BK1_amx bw_test amx_pipe accel_gemm *.o
