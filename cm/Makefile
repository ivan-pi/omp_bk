# Makefile -- the C for Metal kernels (one binary per order) and their SYCL
# hosts.  See README.md in this directory.
#
#   make -C cm                    every kernel and both hosts -> build/cm/
#   make -C cm bk5 ORDERS="3 4"   one kernel, some orders
#   make -C cm check              run both hosts at every order (needs the GPU)
#
# Kernel <k>_cm_<p>[_e<n>].bin: cmc (intel-igc-cm) -> SPIR-V -> the ocloc of
# the same driver release, for GPU_DEVICE with the vector-compute backend;
# nq = p + 2 (bk1), p + 1 (bk5).  A cmc's SPIR-V is finalized only by the IGC
# it was released with; the binary then runs on newer drivers too.
# From nq = 6 a kernel uses the large register file (-double-GRF: 256 GRFs,
# half the threads per EU) to hold its element without spills.  _e<n>: n
# elements per thread (EPT), run with BK_EPT=n; EPT_BUILDS lists the ones
# that are the best configuration of their order.
#
# Host <k>_cm: icpx -fsycl with OpenMP for the reference, -fp-model=precise
# (icpx defaults to a fast model), linked to Level Zero and OpenCL for the
# interop that loads the binaries.  An RPATH to the oneAPI runtime and a
# direct link of libumf (the SYCL adapters dlopen'ed by the runtime need it,
# but ignore the host's RPATH) let the hosts run without the oneAPI
# environment.
#
# Variables (make VAR=value):
#   ORDERS      orders p to build              1 2 3 4 5 6 7 8
#   EPT_BUILDS  kernel:p:ept extra binaries     bk1:1:8 bk1:3:2 bk5:1:8 bk5:2:2
#   GPU_DEVICE  ocloc device                   acm-g10 (Arc A770/A750)
#   GRF         128 or 256 for every order     by nq
#   CMC_FLAGS   extra cmc options
#   KEEP=1      keep the IGC dumps (Gen asm) in $(BUILD_DIR)/dump_<binary>/
#   CMC, OCLOC  the CM compiler and its matched ocloc   $(CM_TOOLS)/cmc, ocloc-cm
#   CM_TOOLS    their directory (README.md: install)    ~/opt/intel-cm-1146/bin
#   ONEAPI_ROOT, BUILD_DIR                      install and output paths
# A change of GPU_DEVICE, GRF, CMC_FLAGS or KEEP needs `make -B`.

CM_TOOLS    ?= $(HOME)/opt/intel-cm-1146/bin
CMC         ?= $(CM_TOOLS)/cmc
OCLOC       ?= $(CM_TOOLS)/ocloc-cm
ONEAPI_ROOT ?= /opt/intel/oneapi
BUILD_DIR   ?= $(abspath $(CURDIR)/../build/cm)
GPU_DEVICE  ?= acm-g10
ORDERS      ?= 1 2 3 4 5 6 7 8
EPT_BUILDS  ?= bk1:1:8 bk1:3:2 bk5:1:8 bk5:2:2
GRF         ?=
CMC_FLAGS   ?=
KEEP        ?= 0

.DEFAULT_GOAL := all

ICPX   := $(realpath $(ONEAPI_ROOT)/compiler/latest)/bin/icpx
UMF    := $(realpath $(ONEAPI_ROOT)/umf/latest)
RPATH  := $(dir $(ICPX))../lib$(if $(UMF),:$(UMF)/lib)
INC    := ../include
B      := $(BUILD_DIR)

, := ,
HOST_FLAGS := -fsycl -std=c++17 -O2 -fp-model=precise -fiopenmp -Wall -I$(INC) -I.
HOST_LIBS  := -lze_loader -lOpenCL $(if $(UMF),-L$(UMF)/lib -Wl$(,)--no-as-needed -lumf -Wl$(,)--as-needed) \
              -Wl,--disable-new-dtags,-rpath,$(RPATH)

# nq of kernel $(1) at order $(2); the -double-GRF option of that nq
nq   = $(word $(2),$(if $(filter bk1,$(1)),3 4 5 6 7 8 9 10,2 3 4 5 6 7 8 9))
grf  = $(if $(GRF),$(filter 256,$(GRF)),$(if $(filter 6 7 8 9 10,$(1)),256))
# binary name of kernel $(1), order $(2), ept $(3)
name = $(1)_cm_$(2)$(if $(filter-out 1,$(3)),_e$(3))

# kernel $(1), order $(2), ept $(3) -> $(B)/<name>.bin
define kernel_rule
$(B)/$(call name,$(1),$(2),$(3)).bin: $(1)_cm_genx.cpp cm_common.h | $(B)
	$(CMC) -march=$(GPU_DEVICE) -emit-spirv -DNQ=$(call nq,$(1),$(2)) -DEPT=$(3) $(CMC_FLAGS) \
	    -o $(B)/$(call name,$(1),$(2),$(3)).spv $$<
	$(if $(filter 1,$(KEEP)),rm -rf $(B)/dump_$(call name,$(1),$(2),$(3)) && \
	    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$(B)/dump_$(call name,$(1),$(2),$(3))) \
	$(OCLOC) compile -q -file $(B)/$(call name,$(1),$(2),$(3)).spv -spirv_input -device $(GPU_DEVICE) \
	    -options "-vc-codegen$(if $(call grf,$(call nq,$(1),$(2))), -double-GRF)" \
	    -output $(call name,$(1),$(2),$(3)) -output_no_suffix -out_dir $(B) \
	    > $(B)/$(call name,$(1),$(2),$(3)).log 2>&1 || { cat $(B)/$(call name,$(1),$(2),$(3)).log; exit 1; }
	@spill=$$$$(sed -n 's/.*Spill memory used = \([0-9]*\) bytes for kernel $(1)_cm.*/\1/p' \
	    $(B)/$(call name,$(1),$(2),$(3)).log); \
	echo "$(call name,$(1),$(2),$(3)).bin  nq=$(call nq,$(1),$(2))$$$${spill:+  SPILL=$$$${spill}B}"
endef

KERNELS := bk1 bk5
ept_of  = $(word 3,$(subst :, ,$(1)))
$(foreach k,$(KERNELS),$(foreach p,$(ORDERS),$(eval $(call kernel_rule,$(k),$(p),1))))
$(foreach x,$(EPT_BUILDS),$(eval $(call kernel_rule,$(word 1,$(subst :, ,$(x))),$(word 2,$(subst :, ,$(x))),$(call ept_of,$(x)))))

# every binary of kernel $(1)
bins = $(foreach p,$(ORDERS),$(B)/$(call name,$(1),$(p),1).bin) \
       $(foreach x,$(filter $(1):%,$(EPT_BUILDS)),$(B)/$(call name,$(1),$(word 2,$(subst :, ,$(x))),$(call ept_of,$(x))).bin)

$(B)/%_cm: %_cm.cpp cm_sycl.hpp $(wildcard $(INC)/bk_*.h) | $(B)
	$(ICPX) $(HOST_FLAGS) $< $(HOST_LIBS) -o $@

$(B):
	mkdir -p $@

.PHONY: all bk1 bk5 check clean
all: bk1 bk5
bk1: $(call bins,bk1) $(B)/bk1_cm
bk5: $(call bins,bk5) $(B)/bk5_cm

# correctness: every order of both kernels (one element per thread) at the
# hosts' default size and work-group sizes; stops at the first wrong result.
# The default sizes are too small for meaningful rates at p = 1, 2.
check: all
	@for p in $(ORDERS); do \
	    for k in bk1 bk5; do \
	        out=$$($(B)/$${k}_cm $$p) || { echo "$$out"; exit 1; }; \
	        echo "$$out" | grep "$${k}_cm_e"; \
	    done; \
	done

clean:
	rm -rf $(B)
