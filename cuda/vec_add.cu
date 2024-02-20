#include "vec_add.cuh"
#include <stdexcept>

namespace Numeric::CUDA
{
__global__ void vec_add_kernels(float *first, float *sec, float *res,
                                unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        res[idx] = first[idx] + sec[idx];
    }
}

std::vector<float> vecAdd(std::vector<float> const &first_host,
                          std::vector<float> const &sec_host)
{
    if (first_host.size() != sec_host.size()) {
        throw std::invalid_argument{"Size should be equal"};
    }
    auto res_host = std::vector<float>(first_host.size());
    auto const vec_size_bytes = first_host.size() * sizeof(float);

    // Allocate device memory.
    auto first_dev = static_cast<float*>(nullptr);
    auto sec_dev = static_cast<float*>(nullptr);
    auto res_dev = static_cast<float*>(nullptr);

    cudaMalloc(reinterpret_cast<void **>(&first_dev), vec_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&sec_dev), vec_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&res_dev), vec_size_bytes);

    // Transfer data from the host to the device.
    cudaMemcpy(first_dev, first_host.data(), vec_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sec_dev, sec_host.data(), vec_size_bytes, cudaMemcpyHostToDevice);

    // Execute the kernel.
    auto const num_threads_per_block = 32u;
    auto const num_blocks = std::ceil(static_cast<float>(first_host.size())/num_threads_per_block);
    vec_add_kernels<<<num_blocks, num_threads_per_block>>>(
        first_dev, sec_dev, res_dev, static_cast<unsigned >(first_host.size()));

    // Transfer result data from the device to host.
    cudaMemcpy(res_host.data(), res_dev, vec_size_bytes, cudaMemcpyDeviceToHost);

    // Cleanup device memory.
    cudaFree(first_dev);
    cudaFree(sec_dev);
    cudaFree(res_dev);

    return res_host;
}

}// Numeric namespace.

