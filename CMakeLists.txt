cmake_minimum_required(VERSION 3.25)

if (NOT DEFINED $ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
endif ()

if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
endif ()

project(pmpp_cuda CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/dlib)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
enable_testing()
add_library(pmpp_cuda_lib
        pmpp/02_vec_add/vec_add.h
        pmpp/02_vec_add/vec_add.cpp
        pmpp/02_vec_add/vec_add.cu
        pmpp/02_vec_add/vec_add.cuh
        pmpp/utils/eigen_utils.h
        pmpp/03_05_06_mat_mul/mat_mul.cuh
        pmpp/03_05_06_mat_mul/mat_mul.cu
        pmpp/utils/dev_config.cuh
        pmpp/utils/dev_config.cu
        pmpp/utils/host_timer.h
        pmpp/utils/host_timer.cpp
        pmpp/utils/exec_config.cuh
        pmpp/07_convolution/conv_1d.cuh
        pmpp/07_convolution/conv_1d.cu
        pmpp/utils/check_error.cuh
        pmpp/utils/check_error.cu
        pmpp/utils/dev_timer.cuh
        pmpp/utils/dev_timer.cu
        pmpp/08_stencil/stencil_1d.cuh
        pmpp/08_stencil/stencil_1d.cu
        pmpp/08_stencil/stencil_2d.cuh
        pmpp/08_stencil/stencil_2d.cu
        pmpp/07_convolution/conv_2d.cuh
        pmpp/07_convolution/conv_2d.cu
        pmpp/09_histogram/histogram.cuh
        pmpp/09_histogram/histogram.cu
        pmpp/10_reduction/sum.cuh
        pmpp/10_reduction/sum.cu
        pmpp/utils/dev_vector.cuh
        pmpp/11_scan/cum_sum_host.h
        pmpp/11_scan/cum_sum_host.cpp
        pmpp/11_scan/cum_sum_dev.cuh
        pmpp/11_scan/cum_sum_dev.cu
        pmpp/03_color_gray_scale/color_gray_scale.cuh
        pmpp/03_color_gray_scale/color_gray_scale.cu
        pmpp/03_05_06_mat_mul/mat_mul_naive.cuh
        pmpp/03_05_06_mat_mul/mat_mul_impl.cuh
        pmpp/03_05_06_mat_mul/mat_mul_cublas.cuh
        pmpp/03_05_06_mat_mul/mat_mul_tiled.cuh
        pmpp/03_05_06_mat_mul/sq_mat_mul_tiled_static_sm.cuh
        pmpp/utils/perf.cuh
        pmpp/03_05_06_mat_mul/sq_mat_mul_tiled_dynamic_sm.cuh
        pmpp/utils/dev_vector_factory.cuh
        pmpp/utils/host_dev_copy.cuh
        pmpp/utils/dlib_utils.cuh
        best_practices_nvidia/mem_optim/copy_execute_latency.cuh
        best_practices_nvidia/mem_optim/mem_bandwidth.cuh
        best_practices_nvidia/mem_optim/mem_bandwidth.cu
        best_practices_nvidia/mem_optim/copy_execute_latency.cu
        best_practices_nvidia/mat_transpose/transpose.cuh
        best_practices_nvidia/mat_transpose/transpose.cu
        best_practices_nvidia/mat_transpose/trans_impl_strategy.cuh
        best_practices_nvidia/mat_transpose/trans_impl_naive.cuh
        best_practices_nvidia/mat_transpose/trans_impl_sm.cuh
        best_practices_nvidia/mat_transpose/trans_impl_sm_no_bank_conflict.cuh

)
target_include_directories(pmpp_cuda_lib PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(pmpp_cuda_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(pmpp_cuda_lib cublas)
target_link_libraries(pmpp_cuda_lib dlib)
add_executable(pmpp_cuda main.cu

)
set_target_properties(pmpp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(pmpp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -ftemplate-backtrace-limit=0 >)
target_link_libraries(pmpp_cuda pmpp_cuda_lib)
target_include_directories(pmpp_cuda PRIVATE ${EIGEN3_INCLUDE_DIR})


add_executable(pmpp_cuda_test main_test.cpp
        pmpp/02_vec_add/vec_add_test.cpp
        pmpp/02_vec_add/vec_add_cuda_test.cpp
        pmpp/utils/eigen_utils_test.cpp
        pmpp/07_convolution/conv_1d_test.cpp
        pmpp/08_stencil/stencil_1d_test.cpp
        pmpp/08_stencil/stencil_2d_test.cpp
        pmpp/07_convolution/conv_2d_test.cpp
        pmpp/09_histogram/histogram_test.cpp
        pmpp/10_reduction/sum_test.cpp
        pmpp/11_scan/cum_sum_host_test.cpp
        pmpp/11_scan/cum_sum_dev_test.cu
        pmpp/03_05_06_mat_mul/mat_mul_test.cu
        best_practices_nvidia/mat_transpose/tranpose_test.cu
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