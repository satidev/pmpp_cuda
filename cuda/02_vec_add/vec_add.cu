#include "vec_add.cuh"
#include "../utils/exec_config.cuh"
#include <stdexcept>
#include "../utils/check_error.cuh"
#include "../utils/timer.cuh"

namespace Numeric::CUDA
{
__global__ void vecAddKernel(float const *first, float const *sec,
                             float *res, unsigned num_elems);

__global__ void vecAddKernel(float const *first, float const *sec,
                             float *res, unsigned num_elems)
{
    auto const idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < num_elems) {
        res[idx] = first[idx] + sec[idx];
    }
}

std::vector<float> vecAdd(std::vector<float> const &first_host,
                          std::vector<float> const &sec_host,
                          bool print_kernel_time)
{
    if (std::size(first_host) != std::size(sec_host)) {
        throw std::invalid_argument{"Vector size should be equal\n"};
    }
    auto const num_elems{static_cast<unsigned>(std::size(first_host))};
    auto res_host{std::vector<float>(num_elems)};
    auto const vec_size_bytes{num_elems * sizeof(float)};

    // Allocate device memory.
    auto first_dev{static_cast<float *>(nullptr)};
    auto sec_dev{static_cast<float *>(nullptr)};
    auto res_dev{static_cast<float *>(nullptr)};

    checkError(cudaMalloc(reinterpret_cast<void **>(&first_dev), vec_size_bytes),
               "allocation of device buffer for first vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&sec_dev), vec_size_bytes),
               "allocation of device buffer for second vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&res_dev), vec_size_bytes),
               "allocation of device buffer for results");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(first_dev, std::data(first_host), vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the first vector to the device");
    checkError(cudaMemcpy(sec_dev, std::data(sec_host), vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the second vector to the device");

    // Execute the kernel.
    auto const exec_params{ExecConfig::getParams(num_elems, vecAddKernel, 0u)};
    auto timer{Timer{}};
    timer.tic();

    vecAddKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        first_dev, sec_dev, res_dev, num_elems);
    checkErrorKernel("vector addition kernel", true);

    auto time_taken_sec{timer.toc()};
    if (print_kernel_time) {
        std::cout << "Time taken (kernel:vec_add):: " << time_taken_sec << " seconds." << std::endl;
    }

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(res_host), res_dev, vec_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    // Cleanup device memory.
    checkError(cudaFree(first_dev), "free the device memory of first vector");
    checkError(cudaFree(sec_dev), "free the device memory of second vector");
    checkError(cudaFree(res_dev), "free the device memory of result vector");

    return res_host;
}

}// Numeric namespace.

