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
#       BK1_<p> BK3_<p> BK1_ff_<p> BK5_<nq>   openmp/ drivers, OpenMP offload
#       bk1_offload_<nq> bk5_offload_<nq>      openmp/offload/ kernel variants
#       bk1_sycl_<nq> bk5_sycl_<nq>            sycl/ (_jit: JIT-compiled, for
#                                              specialization constants)
#       all_<prog>                             every order of <prog>
#     and in `all`: stream_omp (OpenMP offload) and bk_tinytc (if tinytc is found).
#
# The drivers in openmp/ instantiate every order in a switch; the GPU builds
# compile a copy with only the requested `case` (cmake/keep_case.cmake), so
# the device compiler sees one kernel set at a time.  Compiles and links of
# the GPU targets run in the job pool bk_gpu (Ninja; BK_GPU_JOBS at a time).
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
set(BK_ORDERS_BK5 "2;3;4;5;6;7;8" CACHE STRING "nq of BK5_<nq>")
set(BK_ORDERS_BK1_GPU "3;4;5;6;7;8;9;10" CACHE STRING "nq of bk1_offload_<nq>, bk1_sycl_<nq>")
set(BK_ORDERS_BK5_GPU "2;3;4;5;6;7;8;9" CACHE STRING "nq of bk5_offload_<nq>, bk5_sycl_<nq>")
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
        target_include_directories(${name} PRIVATE ${BK_INCLUDE} ${BK_OPENMP} ${CMAKE_SOURCE_DIR}/amx)
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
# run time), sycl_rt (plain SYCL host code, for run-time compiled kernels)
function(bk_gpu_target name api)
    target_include_directories(${name} PRIVATE ${BK_INCLUDE} ${BK_OPENMP})
    if(api STREQUAL "omp")
        target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64_gen)
        target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64_gen
                            "SHELL:-Xopenmp-target-backend=spir64_gen \"-device ${BK_GPU_DEVICE}\"")
    elseif(api STREQUAL "sycl")
        target_compile_options(${name} PRIVATE -fsycl -fsycl-targets=spir64_gen)
        target_link_options(${name} PRIVATE -fsycl -fsycl-targets=spir64_gen
                            "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${BK_GPU_DEVICE}\"")
    elseif(api STREQUAL "sycl_jit")
        target_compile_options(${name} PRIVATE -fsycl -fsycl-targets=spir64)
        target_link_options(${name} PRIVATE -fsycl -fsycl-targets=spir64)
    else()
        target_compile_options(${name} PRIVATE -fsycl)
        target_link_options(${name} PRIVATE -fsycl)
    endif()
    target_compile_options(${name} PRIVATE ${bk_extra})
    target_link_options(${name} PRIVATE ${bk_extra})
    set_target_properties(${name} PROPERTIES JOB_POOL_COMPILE bk_gpu JOB_POOL_LINK bk_gpu)
endfunction()

# The A770 has no fp64 hardware: BK1_ff (double reference, float-float
# split/join) is compiled with IGC's FP64 emulation.
function(bk_dp_emulation name)
    set(launch env IGC_EnableDPEmulation=1 ${CMAKE_CXX_COMPILER_LAUNCHER})
    set(link env IGC_EnableDPEmulation=1 ${CMAKE_CXX_LINKER_LAUNCHER})
    set_target_properties(${name} PROPERTIES CXX_COMPILER_LAUNCHER "${launch}" CXX_LINKER_LAUNCHER "${link}")
endfunction()

# openmp/<prog>.cpp, one order per binary: <prog>_<order>
function(bk_gpu_driver prog)
    add_custom_target(all_${prog})
    foreach(order IN LISTS ARGN)
        set(name ${prog}_${order})
        set(src ${CMAKE_BINARY_DIR}/src/${name}.cpp)
        add_custom_command(OUTPUT ${src}
            COMMAND ${CMAKE_COMMAND} -DIN=${BK_OPENMP}/${prog}.cpp -DOUT=${src} -DORDER=${order}
                    -P ${CMAKE_SOURCE_DIR}/cmake/keep_case.cmake
            DEPENDS ${BK_OPENMP}/${prog}.cpp ${CMAKE_SOURCE_DIR}/cmake/keep_case.cmake
            COMMENT "${prog}.cpp with only case ${order}")
        add_executable(${name} EXCLUDE_FROM_ALL ${src})
        bk_gpu_target(${name} omp)
        if(prog STREQUAL "BK1_ff")
            bk_dp_emulation(${name})
        endif()
        add_dependencies(all_${prog} ${name})
    endforeach()
endfunction()

# <src> compiled once per nq (-DNQ=<nq>): <prog>_<nq>
function(bk_gpu_nq prog src api)
    add_custom_target(all_${prog})
    foreach(nq IN LISTS ARGN)
        set(name ${prog}_${nq})
        add_executable(${name} EXCLUDE_FROM_ALL ${src})
        target_compile_definitions(${name} PRIVATE NQ=${nq})
        bk_gpu_target(${name} ${api})
        add_dependencies(all_${prog} ${name})
    endforeach()
endfunction()

bk_gpu_driver(BK1 ${BK_ORDERS_P})
bk_gpu_driver(BK3 ${BK_ORDERS_P})
bk_gpu_driver(BK1_ff ${BK_ORDERS_P})
bk_gpu_driver(BK5 ${BK_ORDERS_BK5})
bk_gpu_nq(bk1_offload openmp/offload/bk1_offload.cpp omp ${BK_ORDERS_BK1_GPU})
bk_gpu_nq(bk5_offload openmp/offload/bk5_offload.cpp omp ${BK_ORDERS_BK5_GPU})
bk_gpu_nq(bk1_sycl sycl/bk1_sycl.cpp sycl ${BK_ORDERS_BK1_GPU})
bk_gpu_nq(bk5_sycl sycl/bk5_sycl.cpp sycl ${BK_ORDERS_BK5_GPU})

# JIT builds of the SYCL kernels: bk1_sycl_<nq>_jit, bk5_sycl_<nq>_jit
foreach(prog bk1_sycl bk5_sycl)
    if(prog STREQUAL "bk1_sycl")
        set(orders ${BK_ORDERS_BK1_GPU})
    else()
        set(orders ${BK_ORDERS_BK5_GPU})
    endif()
    foreach(nq IN LISTS orders)
        add_executable(${prog}_${nq}_jit EXCLUDE_FROM_ALL sycl/${prog}.cpp)
        target_compile_definitions(${prog}_${nq}_jit PRIVATE NQ=${nq})
        bk_gpu_target(${prog}_${nq}_jit sycl_jit)
    endforeach()
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
