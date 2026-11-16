# omp_bk -- CEED bake-off kernels BK1, BK3, BK5 in OpenMP, SYCL and tinytc.
#
# Two kinds of build tree, chosen by BK_GPU_DEVICE (see CMakePresets.json):
#
#   host (BK_GPU_DEVICE empty, any C++17 compiler; preset "host"):
#     BK1 BK3 BK5 BK1_ff         openmp/      every order in one binary
#     BK1_amx                    amx/         software-emulated AMX off Apple Silicon
#     bw_test amx_pipe stream_omp roofline/   (accel_gemm on macOS)
#   all of them threaded with OpenMP if BK_ENABLE_OPENMP and the compiler
#   supports it, serial otherwise.
#
#   Intel GPU (icpx, BK_GPU_DEVICE = an ocloc device such as acm-g10, or jit
#   for SPIR-V compiled at run time; presets "gpu-openmp", "gpu-sycl" and
#   "a770"): one binary per order, ahead-of-time compiled for BK_GPU_DEVICE,
#   by backend:
#     BK_ENABLE_OPENMP (ON):  OpenMP offload
#       BK1_<p> BK3_<p> BK1_ff_<p> BK5_<p>    openmp/ drivers (-DBK_ORDER=p)
#       bk{1,3,5}_offload_<p>                  openmp/offload/ kernel variants (-DNQ=nq)
#       stream_omp                             roofline/ bandwidth test
#     BK_ENABLE_SYCL (ON):    SYCL
#       bk{1,3,5}_sycl_<p>                     sycl/ (-DNQ=nq; bk1 and bk5 _jit: JIT-compiled,
#                                              for specialization constants)
#     BK_ENABLE_TINYTC (OFF): bk_tinytc        tinytc/ (needs tinytc_sycl, TINYTC_ROOT,
#                                              and BK_ENABLE_SYCL)
#   named by the polynomial order p (BK1, BK3: nq = p + 2, BK5: nq = p + 1),
#     all_<prog>                               every order of <prog>
#     all_openmp, all_sycl                     every program of a backend
#     drivers                                  every order of BK1, BK3, BK5, BK1_ff
#   The per-order programs are in `all` unless BK_GPU_IN_ALL=OFF (preset
#   "a770", where scripts/build.sh builds them by name or through the all_*
#   targets).  BK_ORDERS_* trim the orders.
#   OpenMP and SYCL both come with icpx; tinytc is a separate install.
#
# One order per binary, so the device compiler sees one kernel set at a
# time.  With BK_GPU_JOBS > 0 the compiles and links of the GPU targets run
# in the Ninja job pool bk_gpu, BK_GPU_JOBS at a time (each peaks at about
# 0.5 GB once the host loop optimizer is off, BK_GPU_HOST_LOOPOPT_OFF).
#
# No build type is needed: the flags below include -O3, and the benchmark
# keeps its asserts (no -DNDEBUG).

cmake_minimum_required(VERSION 3.21...3.31)
project(omp_bk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BK_GPU_DEVICE "" CACHE STRING "ocloc device for the Intel GPU targets (e.g. acm-g10), or jit: SPIR-V compiled at run time; empty: host build")
set(BK_GPU_JOBS 0 CACHE STRING "GPU compiles and links run at the same time (Ninja job pool); 0: no limit beyond ninja -j")
set(BK_EXTRA_FLAGS "" CACHE STRING "extra compile and link flags of the GPU targets")
set(BK_ORDERS_P "1;2;3;4;5;6;7;8" CACHE STRING "orders p of BK1_<p>, BK3_<p>, BK1_ff_<p>")
set(BK_ORDERS_BK5 "1;2;3;4;5;6;7" CACHE STRING "orders p of BK5_<p>")
set(BK_ORDERS_GPU "1;2;3;4;5;6;7;8" CACHE STRING "orders p of the GPU variant programs (bk1_offload_<p>, bk3_sycl_<p>, ...)")
option(BK_ENABLE_OPENMP "OpenMP: threads in the host tree, offload programs in the GPU tree" ON)
option(BK_ENABLE_SYCL "SYCL programs (GPU tree)" ON)
option(BK_ENABLE_TINYTC "bk_tinytc (GPU tree; needs tinytc and BK_ENABLE_SYCL)" OFF)
option(BK_GPU_HOST_LOOPOPT_OFF "GPU tree: compile host code without icpx's loop optimizer (much faster builds, same device code)" ON)
option(BK_GPU_IN_ALL "put the per-order GPU programs in `all` (each takes minutes to build)" ON)

if(CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
    message(WARNING "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} defines NDEBUG; the benchmark is meant to keep its asserts")
endif()

# Native ISA: -march=native on x86 (where -mcpu= is only a tuning alias and
# would leave std::fma a library call), -mcpu=native on AArch64 (Apple clang
# does not accept -march=native).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
    add_compile_options(-march=native)
else()
    add_compile_options(-mcpu=native)
endif()
add_compile_options(-Wall -O3)
# icpx defaults to a fast floating-point model, which breaks the error-free
# transformations of float_float.h.  Never -ffast-math.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-fp-model=precise)
endif()

set(BK_INCLUDE ${CMAKE_SOURCE_DIR}/include)
set(BK_OPENMP ${CMAKE_SOURCE_DIR}/openmp)

if(NOT BK_GPU_DEVICE)
    # ---------------------------------------------------------------- host
    add_compile_options(-pedantic)
    if(BK_ENABLE_OPENMP)
        find_package(OpenMP)
        if(NOT OpenMP_CXX_FOUND)
            message(WARNING "${CMAKE_CXX_COMPILER_ID} has no OpenMP: the host programs run serially (BK_ENABLE_OPENMP=OFF silences this)")
        endif()
    endif()
    if(OpenMP_CXX_FOUND)
        message(STATUS "host tree: OpenMP threads (SYCL and tinytc programs need a GPU tree, BK_GPU_DEVICE)")
    else()
        message(STATUS "host tree: serial, no OpenMP (SYCL and tinytc programs need a GPU tree, BK_GPU_DEVICE)")
    endif()

    function(bk_host name src)
        add_executable(${name} ${src})
        target_include_directories(${name} PRIVATE ${BK_INCLUDE} ${BK_OPENMP})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endfunction()

    bk_host(BK1 openmp/BK1.cpp)
    bk_host(BK3 openmp/BK3.cpp)
    bk_host(BK5 openmp/BK5.cpp)
    # float-float BK1; -DFF_PLANE_BATCH=n in CMAKE_CXX_FLAGS sets the
    # elements per team of the plane kernel
    bk_host(BK1_ff openmp/BK1_ff.cpp)
    bk_host(BK1_amx amx/BK1_amx.cpp)
    bk_host(bw_test roofline/bw_test.cpp)
    bk_host(amx_pipe roofline/amx_pipe.cpp)
    # amx.h
    target_include_directories(bw_test PRIVATE ${CMAKE_SOURCE_DIR}/amx)
    target_include_directories(amx_pipe PRIVATE ${CMAKE_SOURCE_DIR}/amx)
    # host fallback of the GPU bandwidth test
    bk_host(stream_omp roofline/stream_omp.cpp)
    if(APPLE)
        bk_host(accel_gemm roofline/accel_gemm.cpp)
        target_compile_definitions(accel_gemm PRIVATE ACCELERATE_NEW_LAPACK)
        target_link_libraries(accel_gemm PRIVATE "-framework Accelerate")
    endif()
    return()
endif()

# ------------------------------------------------------------------ Intel GPU
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(FATAL_ERROR "BK_GPU_DEVICE=${BK_GPU_DEVICE} needs icpx (CMAKE_CXX_COMPILER=icpx), not ${CMAKE_CXX_COMPILER_ID}")
endif()
if(BK_GPU_JOBS GREATER 0)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS bk_gpu=${BK_GPU_JOBS})
endif()
separate_arguments(bk_extra UNIX_COMMAND "${BK_EXTRA_FLAGS}")

# api: omp (OpenMP offload, AOT), sycl (AOT), sycl_jit (SPIR-V, compiled at
# run time), sycl_rt (plain SYCL host code, for run-time compiled kernels).
# The SYCL programs use host OpenMP for their references (-fiopenmp, no
# offload targets).
function(bk_gpu_target name api)
    target_include_directories(${name} PRIVATE ${BK_INCLUDE})
    if(BK_GPU_DEVICE STREQUAL "jit" AND api STREQUAL "omp")
        target_include_directories(${name} PRIVATE ${BK_OPENMP})
        target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64)
        target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64)
    elseif(BK_GPU_DEVICE STREQUAL "jit" AND api STREQUAL "sycl")
        bk_gpu_target(${name} sycl_jit)
        return()
    elseif(api STREQUAL "omp")
        target_include_directories(${name} PRIVATE ${BK_OPENMP})
        target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64_gen)
        target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64_gen
                            "SHELL:-Xopenmp-target-backend=spir64_gen \"-device ${BK_GPU_DEVICE}\"")
    elseif(api STREQUAL "sycl")
        target_compile_options(${name} PRIVATE -fsycl -fsycl-targets=spir64_gen -fiopenmp)
        target_link_options(${name} PRIVATE -fsycl -fsycl-targets=spir64_gen -fiopenmp
                            "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${BK_GPU_DEVICE}\"")
    elseif(api STREQUAL "sycl_jit")
        target_compile_options(${name} PRIVATE -fsycl -fsycl-targets=spir64 -fiopenmp)
        target_link_options(${name} PRIVATE -fsycl -fsycl-targets=spir64 -fiopenmp)
    else()
        target_compile_options(${name} PRIVATE -fsycl -fiopenmp)
        target_link_options(${name} PRIVATE -fsycl -fiopenmp)
    endif()
    # icpx's loop optimizer (HIR) spends almost all of the compile time of these
    # programs on the host side (BK3_4: 81 s of 82 s); -loopopt=0 reaches only
    # the host compile jobs (as -floopopt-pipeline=none), so the device code is
    # unchanged and only the host fallback and the references lose HIR.
    if(BK_GPU_HOST_LOOPOPT_OFF)
        target_compile_options(${name} PRIVATE "SHELL:-mllvm -loopopt=0")
    endif()
    target_compile_options(${name} PRIVATE ${bk_extra})
    target_link_options(${name} PRIVATE ${bk_extra})
    if(BK_GPU_JOBS GREATER 0)
        set_target_properties(${name} PROPERTIES JOB_POOL_COMPILE bk_gpu JOB_POOL_LINK bk_gpu)
    endif()
endfunction()

# <src> once per order p, -D<define>=<p + offset>: targets <prog>_<p>[<suffix>]
# and all_<prog>[<suffix>], the latter added to all_<backend> (openmp, sycl)
if(BK_GPU_IN_ALL)
    set(bk_exclude "")
else()
    set(bk_exclude EXCLUDE_FROM_ALL)
endif()
add_custom_target(all_openmp)
add_custom_target(all_sycl)
function(bk_gpu_orders prog src api define offset suffix)
    add_custom_target(all_${prog}${suffix})
    if(api STREQUAL "omp")
        add_dependencies(all_openmp all_${prog}${suffix})
    else()
        add_dependencies(all_sycl all_${prog}${suffix})
    endif()
    foreach(order IN LISTS ARGN)
        set(name ${prog}_${order}${suffix})
        math(EXPR value "${order} + ${offset}")
        add_executable(${name} ${bk_exclude} ${src})
        target_compile_definitions(${name} PRIVATE ${define}=${value})
        bk_gpu_target(${name} ${api})
        add_dependencies(all_${prog}${suffix} ${name})
    endforeach()
endfunction()

if(NOT BK_ENABLE_OPENMP AND NOT BK_ENABLE_SYCL)
    message(WARNING "GPU tree with BK_ENABLE_OPENMP=OFF and BK_ENABLE_SYCL=OFF: no BK programs")
endif()

if(BK_ENABLE_OPENMP)
    # drivers: every order of the openmp/ drivers, without the kernel variants
    bk_gpu_orders(BK1 openmp/BK1.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
    bk_gpu_orders(BK3 openmp/BK3.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
    bk_gpu_orders(BK1_ff openmp/BK1_ff.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
    bk_gpu_orders(BK5 openmp/BK5.cpp omp BK_ORDER 0 "" ${BK_ORDERS_BK5})
    # drivers: every order of the openmp/ drivers, without the kernel variants
    add_custom_target(drivers)
    add_dependencies(drivers all_BK1 all_BK3 all_BK1_ff all_BK5)
    bk_gpu_orders(bk1_offload openmp/offload/bk1_offload.cpp omp NQ 2 "" ${BK_ORDERS_GPU})
    bk_gpu_orders(bk5_offload openmp/offload/bk5_offload.cpp omp NQ 1 "" ${BK_ORDERS_GPU})
    bk_gpu_orders(bk3_offload openmp/offload/bk3_offload.cpp omp NQ 2 "" ${BK_ORDERS_GPU})

    # The A770 has no fp64 hardware: BK1_ff (double reference, float-float
    # split/join) is compiled with IGC's FP64 emulation.
    set(dp_compile env IGC_EnableDPEmulation=1 ${CMAKE_CXX_COMPILER_LAUNCHER})
    set(dp_link env IGC_EnableDPEmulation=1 ${CMAKE_CXX_LINKER_LAUNCHER})
    foreach(p IN LISTS BK_ORDERS_P)
        set_target_properties(BK1_ff_${p} PROPERTIES CXX_COMPILER_LAUNCHER "${dp_compile}" CXX_LINKER_LAUNCHER "${dp_link}")
    endforeach()

    add_executable(stream_omp roofline/stream_omp.cpp)
    bk_gpu_target(stream_omp omp)
    add_dependencies(all_openmp stream_omp)
endif()

if(BK_ENABLE_SYCL)
    bk_gpu_orders(bk1_sycl sycl/bk1_sycl.cpp sycl NQ 2 "" ${BK_ORDERS_GPU})
    bk_gpu_orders(bk5_sycl sycl/bk5_sycl.cpp sycl NQ 1 "" ${BK_ORDERS_GPU})
    bk_gpu_orders(bk3_sycl sycl/bk3_sycl.cpp sycl NQ 2 "" ${BK_ORDERS_GPU})
    bk_gpu_orders(bk1_sycl sycl/bk1_sycl.cpp sycl_jit NQ 2 _jit ${BK_ORDERS_GPU})
    bk_gpu_orders(bk5_sycl sycl/bk5_sycl.cpp sycl_jit NQ 1 _jit ${BK_ORDERS_GPU})
endif()

# tinytc: the kernels (tinytc/ir/*.ir) are compiled at run time, so the
# driver is plain SYCL host code.  Optional: asked for, it must be found.
if(BK_ENABLE_TINYTC)
    if(NOT BK_ENABLE_SYCL)
        message(FATAL_ERROR "BK_ENABLE_TINYTC needs BK_ENABLE_SYCL (bk_tinytc is a SYCL program)")
    endif()
    # tinytc installs its own config packages (tinytc_sycl pulls in tinytc,
    # tinytc_ze and tinytc_cl).  The install is found from -DTINYTC_ROOT, else
    # $TINYTC_ROOT (set by tinytc's setvars.sh), else CMAKE_PREFIX_PATH.
    if(NOT TINYTC_ROOT AND DEFINED ENV{TINYTC_ROOT})
        set(TINYTC_ROOT "$ENV{TINYTC_ROOT}")
    endif()
    if(TINYTC_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${TINYTC_ROOT}")
    endif()
    find_package(tinytc_sycl CONFIG)
    if(NOT tinytc_sycl_FOUND)
        message(FATAL_ERROR "BK_ENABLE_TINYTC: tinytc_sycl not found (TINYTC_ROOT=${TINYTC_ROOT}); "
                            "source <tinytc>/setvars.sh, pass -DTINYTC_ROOT=<tinytc install>, "
                            "or set BK_ENABLE_TINYTC=OFF")
    endif()
    message(STATUS "tinytc: ${tinytc_sycl_DIR}")
    add_executable(bk_tinytc tinytc/bk_tinytc.cpp)
    bk_gpu_target(bk_tinytc sycl_rt)
    target_compile_definitions(bk_tinytc PRIVATE TT_IR_DIR="${CMAKE_SOURCE_DIR}/tinytc/ir")
    target_link_libraries(bk_tinytc PRIVATE tinytc::tinytc_sycl)
endif()

if(BK_GPU_IN_ALL)
    set(bk_where "in all")
else()
    set(bk_where "not in all: build them by name, all_<prog>, all_openmp or all_sycl")
endif()
message(STATUS "GPU tree for ${BK_GPU_DEVICE}: OpenMP offload ${BK_ENABLE_OPENMP}, SYCL ${BK_ENABLE_SYCL}, "
               "tinytc ${BK_ENABLE_TINYTC}; per-order programs ${bk_where}")
