#include "vec_add.cuh"
#include "exec_config.cuh"
#include <stdexcept>
#include "check_error.cuh"
#include "timer.cuh"

namespace Numeric::CUDA
{
__global__ void vec_add(float const *first,
                        float const *sec,
                        float *res,
                        unsigned num_elems);

__global__ void vec_add(float const *first,
                        float const *sec,
                        float *res,
                        unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        res[idx] = first[idx] + sec[idx];
    }
}

std::vector<float> vecAdd(std::vector<float> const &first_host,
                          std::vector<float> const &sec_host,
                          bool print_kernel_time)
{
    if (first_host.size() != sec_host.size()) {
        throw std::invalid_argument{"Vector size should be equal\n"};
    }
    auto res_host = std::vector<float>(first_host.size());
    auto const vec_size_bytes = first_host.size() * sizeof(float);

    // Allocate device memory.
    auto first_dev = static_cast<float*>(nullptr);
    auto sec_dev = static_cast<float*>(nullptr);
    auto res_dev = static_cast<float*>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&first_dev), vec_size_bytes),
               "allocation of device buffer for first vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&sec_dev), vec_size_bytes),
               "allocation of device buffer for second vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&res_dev), vec_size_bytes),
               "allocation of device buffer for results");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(first_dev, first_host.data(), vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the first vector to the device");
    checkError(cudaMemcpy(sec_dev, sec_host.data(), vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the second vector to the device");

    // Execute the kernel.
    auto const exec_params = ExecConfig::getParams(first_host.size(), vec_add, 0u);
    auto timer = Timer{};
    timer.tic();
    vec_add<<<exec_params.grid_dim, exec_params.block_dim>>>(
        first_dev, sec_dev, res_dev, static_cast<unsigned >(first_host.size()));
    auto time_taken_sec = timer.toc();
    if(print_kernel_time){
        std::cout << "Time taken (kernel:vec_add):: " << time_taken_sec << " seconds." << std::endl;
    }

    checkErrorKernel("vector addition kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(res_host.data(), res_dev, vec_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    // Cleanup device memory.
    checkError(cudaFree(first_dev), "free the device memory of first vector");
    checkError(cudaFree(sec_dev), "free the device memory of second vector");
    checkError(cudaFree(res_dev), "free the device memory of result vector");

    return res_host;
}

}// Numeric namespace.

