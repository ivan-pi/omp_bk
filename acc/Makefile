# OpenACC builds of the bake-off kernels (acc/BK?_acc.cpp).
#
#   make -C acc                       # nvc++ for the GPU, with the -Minfo=acc report
#   make -C acc ACC=-acc=multicore    # nvc++, OpenACC on the host cores
#   make -C acc host                  # BK?_acc_host: the same sources run serially on
#                                     # the host (nvc++ -acc=host); validate.sh compares
#                                     # the two builds on random data
#   make -C acc CXX=g++               # GCC's OpenACC (-fopenacc): a host fallback
#                                     # without an offload toolchain, used to check that
#                                     # the sources build and run
#
# ACC, INFO and HOSTACC can be overridden on the command line, e.g.
#   make -C acc ACC="-acc=gpu -gpu=cc80"

CXX = nvc++

ifneq ($(findstring nvc,$(CXX)),)
  ACC     ?= -acc=gpu
  INFO    ?= -Minfo=acc
  HOSTACC ?= -acc=host
else
  ACC     ?= -fopenacc
  INFO    ?=
  HOSTACC ?= -Wno-unknown-pragmas
endif

CXXFLAGS = -Wall -O3 -std=c++17

HEADERS = acc_common.h ../bk_common.h

.PHONY: all host clean
all: BK1_acc BK3_acc BK5_acc
host: BK1_acc_host BK3_acc_host BK5_acc_host

BK%_acc: BK%_acc.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(ACC) $(INFO) -o $@ $<

BK%_acc_host: BK%_acc.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(HOSTACC) -o $@ $<

clean:
	$(RM) BK1_acc BK3_acc BK5_acc BK1_acc_host BK3_acc_host BK5_acc_host *.o
