cmake_minimum_required(VERSION 3.25)

if (NOT DEFINED $ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
endif ()

if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
endif ()

project(pmpp_cuda CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
find_package(PythonLibs 3 REQUIRED)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/dlib)
enable_testing()
add_library(cuda_lib
        pmpp/03_05_06_mat_mul/mat_mul.cuh
        pmpp/03_05_06_mat_mul/mat_mul.cu
        utils/dev_config.cuh
        utils/dev_config.cu
        utils/host_timer.h
        utils/host_timer.cpp
        utils/exec_config.cuh
        pmpp/07_convolution/conv_1d.cuh
        pmpp/07_convolution/conv_1d.cu
        utils/check_error.cuh
        utils/check_error.cu
        utils/dev_timer.cuh
        utils/dev_timer.cu
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
        utils/dev_vector.cuh
        pmpp/11_scan/cum_sum_host.h
        pmpp/11_scan/cum_sum_host.cpp
        pmpp/11_scan/cum_sum_dev.cuh
        pmpp/11_scan/cum_sum_dev.cu
        pmpp/03_color_gray_scale/color_gray_scale.cuh
        pmpp/03_color_gray_scale/color_gray_scale.cu
        pmpp/03_05_06_mat_mul/mat_mul_naive.cuh
        pmpp/03_05_06_mat_mul/mat_mul_impl_strategy.cuh
        pmpp/03_05_06_mat_mul/mat_mul_cublas.cuh
        pmpp/03_05_06_mat_mul/mat_mul_tiled.cuh
        pmpp/03_05_06_mat_mul/sq_mat_mul_tiled_static_sm.cuh
        utils/perf.cuh
        pmpp/03_05_06_mat_mul/sq_mat_mul_tiled_dynamic_sm.cuh
        utils/dev_vector_factory.cuh
        utils/host_dev_copy.cuh
        utils/dlib_utils.cuh
        bpnv/mem_optim/copy_execute_latency.cuh
        bpnv/mem_optim/mem_bandwidth.cuh
        bpnv/mem_optim/mem_bandwidth.cu
        bpnv/mem_optim/copy_execute_latency.cu
        bpnv/mat_transpose/transpose.cuh
        bpnv/mat_transpose/transpose.cu
        bpnv/mat_transpose/trans_impl_strategy.cuh
        bpnv/mat_transpose/trans_impl_naive.cuh
        bpnv/mat_transpose/trans_impl_sm.cuh
        bpnv/mat_transpose/trans_impl_sm_padding.cuh
        pmpp/02_vec_add/vec_add_cublas.cuh
        pmpp/02_vec_add/vec_add.cuh
        pmpp/02_vec_add/vec_add.cu
        bpnv/mat_transpose/trans_impl_sm_swizzling.cuh
        bpnv/mat_transpose/perf_test.cuh
        bpnv/mat_transpose/perf_test.cu
        utils/stream_adaptor.cuh
        utils/dev_vector_async.cuh
        utils/pinned_vector.cuh
        utils/mapped_vector.cuh
)
target_include_directories(cuda_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(cuda_lib cublas)
target_link_libraries(cuda_lib dlib)

add_executable(run_perf_tests main.cu
        perf_test_analyzer.h
        perf_test_analyzer.cpp)
set_target_properties(run_perf_tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(run_perf_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -ftemplate-backtrace-limit=0 >)
target_link_libraries(run_perf_tests cuda_lib)
target_include_directories(run_perf_tests PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(run_perf_tests ${PYTHON_LIBRARIES})
target_include_directories(run_perf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/matplotlib-cpp)
add_executable(run_unit_tests main_test.cpp
        pmpp/07_convolution/conv_1d_test.cpp
        pmpp/08_stencil/stencil_1d_test.cpp
        pmpp/08_stencil/stencil_2d_test.cpp
        pmpp/07_convolution/conv_2d_test.cpp
        pmpp/09_histogram/histogram_test.cpp
        pmpp/10_reduction/sum_test.cpp
        pmpp/11_scan/cum_sum_host_test.cpp
        pmpp/11_scan/cum_sum_dev_test.cu
        pmpp/03_05_06_mat_mul/mat_mul_test.cu
        bpnv/mat_transpose/unit_test.cu
        pmpp/02_vec_add/vec_add_test.cu
        utils/dev_vec_async_unit_test.cu
        utils/dev_vector_unit_test.cu
)
set_target_properties(run_unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(run_unit_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 >
)
target_link_libraries(run_unit_tests cuda_lib)
target_link_libraries(run_unit_tests GTest::gtest_main GTest::gmock_main)
include(GoogleTest)
gtest_discover_tests(run_unit_tests)