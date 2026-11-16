# omp_bk -- CEED bake-off kernels BK1, BK3, BK5 in OpenMP, SYCL and tinytc.
#
# Two kinds of build tree, chosen by BK_GPU_DEVICE (see CMakePresets.json):
#
#   host (BK_GPU_DEVICE empty, any C++17 compiler; preset "host"):
#     BK1 BK3 BK5 BK1_ff         openmp/      every order in one binary
#     BK1_amx                    amx/         software-emulated AMX off Apple Silicon
#     bw_test amx_pipe stream_omp roofline/   (accel_gemm on macOS)
#
#   Intel GPU (icpx, BK_GPU_DEVICE = an ocloc device such as acm-g10; preset "a770"):
#     one binary per order, ahead-of-time compiled for BK_GPU_DEVICE, none of
#     them in `all` (each takes minutes and gigabytes to build):
#       BK1_<p> BK3_<p> BK1_ff_<p> BK5_<p>    openmp/ drivers, OpenMP offload (-DBK_ORDER=p)
#       bk{1,3,5}_offload_<p>                  openmp/offload/ kernel variants (-DNQ=nq)
#       bk{1,3,5}_sycl_<p>                     sycl/ (-DNQ=nq; bk1 and bk5 _jit: JIT-compiled,
#                                              for specialization constants)
#     named by the polynomial order p (BK1, BK3: nq = p + 2, BK5: nq = p + 1),
#       all_<prog>                             every order of <prog>
#     and in `all`: stream_omp (OpenMP offload) and bk_tinytc (if tinytc is found).
#
# One order per binary, so the device compiler sees one kernel set at a
# time.  Compiles and links of the GPU targets run in the job pool bk_gpu
# (Ninja; BK_GPU_JOBS at a time).
#
# No build type is needed: the flags below include -O3, and the benchmark
# keeps its asserts (no -DNDEBUG).

cmake_minimum_required(VERSION 3.21)
project(omp_bk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BK_GPU_DEVICE "" CACHE STRING "ocloc device for the Intel GPU targets (e.g. acm-g10); empty: host build")
set(BK_GPU_JOBS 1 CACHE STRING "GPU compiles and links run at the same time (Ninja job pool)")
set(BK_EXTRA_FLAGS "" CACHE STRING "extra compile and link flags of the GPU targets")
set(BK_ORDERS_P "1;2;3;4;5;6;7;8" CACHE STRING "orders p of BK1_<p>, BK3_<p>, BK1_ff_<p>")
set(BK_ORDERS_BK5 "1;2;3;4;5;6;7" CACHE STRING "orders p of BK5_<p>")
set(BK_ORDERS_GPU "1;2;3;4;5;6;7;8" CACHE STRING "orders p of the GPU variant programs (bk1_offload_<p>, bk3_sycl_<p>, ...)")
set(BK_TINYTC_ROOT "/opt/tinytc/latest" CACHE PATH "tinytc install searched for bk_tinytc")

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
    find_package(OpenMP)

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
set_property(GLOBAL APPEND PROPERTY JOB_POOLS bk_gpu=${BK_GPU_JOBS})
separate_arguments(bk_extra UNIX_COMMAND "${BK_EXTRA_FLAGS}")

# api: omp (OpenMP offload, AOT), sycl (AOT), sycl_jit (SPIR-V, compiled at
# run time), sycl_rt (plain SYCL host code, for run-time compiled kernels).
# The SYCL programs use host OpenMP for their references (-fiopenmp, no
# offload targets).
function(bk_gpu_target name api)
    target_include_directories(${name} PRIVATE ${BK_INCLUDE})
    if(api STREQUAL "omp")
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
    target_compile_options(${name} PRIVATE ${bk_extra})
    target_link_options(${name} PRIVATE ${bk_extra})
    set_target_properties(${name} PROPERTIES JOB_POOL_COMPILE bk_gpu JOB_POOL_LINK bk_gpu)
endfunction()

# <src> once per order p, -D<define>=<p + offset>: targets <prog>_<p>[<suffix>]
# and all_<prog>[<suffix>]
function(bk_gpu_orders prog src api define offset suffix)
    add_custom_target(all_${prog}${suffix})
    foreach(order IN LISTS ARGN)
        set(name ${prog}_${order}${suffix})
        math(EXPR value "${order} + ${offset}")
        add_executable(${name} EXCLUDE_FROM_ALL ${src})
        target_compile_definitions(${name} PRIVATE ${define}=${value})
        bk_gpu_target(${name} ${api})
        add_dependencies(all_${prog}${suffix} ${name})
    endforeach()
endfunction()

bk_gpu_orders(BK1 openmp/BK1.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
bk_gpu_orders(BK3 openmp/BK3.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
bk_gpu_orders(BK1_ff openmp/BK1_ff.cpp omp BK_ORDER 0 "" ${BK_ORDERS_P})
bk_gpu_orders(BK5 openmp/BK5.cpp omp BK_ORDER 0 "" ${BK_ORDERS_BK5})
bk_gpu_orders(bk1_offload openmp/offload/bk1_offload.cpp omp NQ 2 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk5_offload openmp/offload/bk5_offload.cpp omp NQ 1 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk3_offload openmp/offload/bk3_offload.cpp omp NQ 2 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk1_sycl sycl/bk1_sycl.cpp sycl NQ 2 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk5_sycl sycl/bk5_sycl.cpp sycl NQ 1 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk3_sycl sycl/bk3_sycl.cpp sycl NQ 2 "" ${BK_ORDERS_GPU})
bk_gpu_orders(bk1_sycl sycl/bk1_sycl.cpp sycl_jit NQ 2 _jit ${BK_ORDERS_GPU})
bk_gpu_orders(bk5_sycl sycl/bk5_sycl.cpp sycl_jit NQ 1 _jit ${BK_ORDERS_GPU})

# The A770 has no fp64 hardware: BK1_ff (double reference, float-float
# split/join) is compiled with IGC's FP64 emulation.
set(dp_compile env IGC_EnableDPEmulation=1 ${CMAKE_CXX_COMPILER_LAUNCHER})
set(dp_link env IGC_EnableDPEmulation=1 ${CMAKE_CXX_LINKER_LAUNCHER})
foreach(p IN LISTS BK_ORDERS_P)
    set_target_properties(BK1_ff_${p} PROPERTIES CXX_COMPILER_LAUNCHER "${dp_compile}" CXX_LINKER_LAUNCHER "${dp_link}")
endforeach()

add_executable(stream_omp roofline/stream_omp.cpp)
bk_gpu_target(stream_omp omp)

# tinytc: the kernels (tinytc/ir/*.ir) are compiled at run time, so the
# driver is plain SYCL host code.
list(APPEND CMAKE_PREFIX_PATH ${BK_TINYTC_ROOT})
find_package(tinytc_sycl CONFIG QUIET)
if(tinytc_sycl_FOUND)
    add_executable(bk_tinytc tinytc/bk_tinytc.cpp)
    bk_gpu_target(bk_tinytc sycl_rt)
    target_compile_definitions(bk_tinytc PRIVATE TT_IR_DIR="${CMAKE_SOURCE_DIR}/tinytc/ir")
    target_link_libraries(bk_tinytc PRIVATE tinytc::tinytc_sycl)
else()
    message(STATUS "tinytc_sycl not found under BK_TINYTC_ROOT=${BK_TINYTC_ROOT}: no bk_tinytc")
endif()
