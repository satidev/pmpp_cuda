cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED $ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
endif()

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wall \
    -Wextra \
    -Werror \
    -Wpedantic \
    -Wconversion \
    -Wdeprecated-declarations \
    -Wformat \
    -Wfloat-equal \
    -Wshadow \
    -Wuninitialized \
    -Wunused \
    -Wmissing-declarations \
    -Wmissing-include-dirs \
    -Wcast-align \
    -Wredundant-decls \
    -Wlogical-op \
    -Wnull-dereference \
    -Wreturn-type \
    -Wsign-conversion \
    -Warith-conversion \
    -Walloc-zero \
    -Wclass-memaccess \
    -Wdouble-promotion \
    -Wduplicated-branches \
    -Wnon-virtual-dtor \
    -Wnull-dereference \
    -Wold-style-cast \
    -Wunreachable-code")


# Exclude third party files from compiler warnings.
set(CMAKE_CXX_FLAGS -isystem"${EIGEN3_INCLUDE_DIR}")


enable_testing()
add_executable(pmpp_cuda main.cpp
        cpp/vec_add.h
        cpp/vec_add.cpp
        cpp/vec_add_test.cpp
        cuda/vec_add.cu
        cuda/vec_add.cuh
        cuda/vec_add_cuda_test.cpp
        cpp/eigen_utils.h
        cpp/eigen_utils_test.cpp
        cuda/mat_mul.cuh
        cuda/mat_mul.cu
        cuda/mat_mul_cuda_test.cpp
)
target_link_libraries(pmpp_cuda GTest::gtest_main)
target_include_directories(pmpp_cuda PRIVATE ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(pmpp_cuda)