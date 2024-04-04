cmake_minimum_required(VERSION 3.25)

if (NOT DEFINED $ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
endif ()

if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
endif ()

project(pmpp_cuda CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
enable_testing()
add_library(pmpp_cuda_lib
        cpp/vec_add.h
        cpp/vec_add.cpp
        cuda/vec_add.cu
        cuda/vec_add.cuh
        cpp/eigen_utils.h
        cuda/mat_mul.cuh
        cuda/mat_mul.cu
        cuda/dev_config.cuh
        cuda/dev_config.cu
        cpp/host_timer.h
        cpp/host_timer.cpp
        cuda/exec_config.cuh
        cuda/conv.cuh
        cuda/conv.cu
        cuda/check_error.cuh
        cuda/check_error.cu
        cuda/timer.cuh
        cuda/timer.cu
        cuda/stencil_1d.cuh
        cuda/stencil_1d.cu
        cuda/stencil_2d.cuh
        cuda/stencil_2d.cu
)
target_include_directories(pmpp_cuda_lib PRIVATE ${EIGEN3_INCLUDE_DIR})


add_executable(pmpp_cuda main.cu)
set_target_properties(pmpp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(pmpp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -ftemplate-backtrace-limit=0 >)
target_link_libraries(pmpp_cuda pmpp_cuda_lib)
target_include_directories(pmpp_cuda PRIVATE ${EIGEN3_INCLUDE_DIR})


add_executable(pmpp_cuda_test main_test.cpp
        cpp/vec_add_test.cpp
        cuda/vec_add_cuda_test.cpp
        cpp/eigen_utils_test.cpp
        cuda/mat_mul_cuda_test.cpp
        cuda/conv_test.cpp
        cuda/stencil_1d_test.cpp
        cuda/stencil_2d_test.cpp
)
set_target_properties(pmpp_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(pmpp_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 >
)
target_link_libraries(pmpp_cuda_test pmpp_cuda_lib)
target_link_libraries(pmpp_cuda_test GTest::gtest_main GTest::gmock_main)
target_include_directories(pmpp_cuda_test PRIVATE ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(pmpp_cuda_test)