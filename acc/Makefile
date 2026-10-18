# OpenACC builds of the bake-off kernels (acc/*.cpp).
#
#   make -C acc                       # nvc++ for the GPU, with the -Minfo=acc report
#   make -C acc ACC=-acc=multicore    # nvc++, OpenACC on the host cores
#   make -C acc host                  # *_host: the same sources run serially on the
#                                     # host (nvc++ -acc=host); validate.sh compares
#                                     # the two builds on random data
#   make -C acc CXX=g++               # GCC's OpenACC (-fopenacc): a host fallback
#                                     # without an offload toolchain, used to check that
#                                     # the sources build and run
#
# ACC, INFO and HOSTACC can be overridden on the command line, e.g.
#   make -C acc ACC="-acc=gpu -gpu=cc80"
# DEFS passes -D options to every build, e.g. the knobs of BK1_acc:
#   make -C acc BK1_acc DEFS="-DBK_EPG=3 -DBK_CACHE_BASIS"

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
DEFS =

HEADERS = acc_common.h ../bk_common.h

# BK1_acc holds several kernels, selected at run time with BK_KERNEL (see
# the file's header); BK3_acc and BK5_acc are one element per gang with the
# inner loops left to the compiler.
PROGS = BK1_acc BK3_acc BK5_acc

.PHONY: all host clean
all: $(PROGS)
host: $(PROGS:=_host)

$(PROGS): %: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(DEFS) $(ACC) $(INFO) -o $@ $<

$(PROGS:=_host): %_host: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(DEFS) $(HOSTACC) -o $@ $<

clean:
	$(RM) $(PROGS) $(PROGS:=_host) *.o
