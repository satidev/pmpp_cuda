#include "stencil_1d.cuh"
#include "../utils/exec_config.cuh"
#include "../utils/check_error.cuh"

namespace Numeric::CUDA
{

__global__ void diff_kernel(float const *ip, float *op,
                            unsigned num_elems_ip)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1 && idx < num_elems_ip) {
        op[idx - 1] = ip[idx] - ip[idx - 1];
    }
}

__global__ void sum_3point_kernel(float const *ip, float *op,
                                  unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1u && idx < (num_elems - 1)) {
        op[idx] = ip[idx - 1] + ip[idx] + ip[idx + 1];
    }
    else if (idx == 0u || idx == (num_elems - 1)) {
        op[idx] = ip[idx];
    }
}

__global__ void sum_3point_sm_kernel(float const *ip, float *op,
                                     unsigned num_elems)
{
    auto const ip_tile_size = static_cast<int>(blockDim.x);
    auto const op_tile_size = ip_tile_size - 2;

    auto const idx = static_cast<int>(blockIdx.x * op_tile_size + threadIdx.x) - 1;

    // Copy to shared memory.
    extern __shared__ float sm[];
    if (idx >= 0 && idx < num_elems) {
        sm[threadIdx.x] = ip[idx];
    }
    else {
        sm[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if(idx >= 0 && idx < num_elems){

        if(idx > 0 && idx < (num_elems - 1)) {
            if(threadIdx.x > 0 && threadIdx.x < (ip_tile_size - 1u)) {
                op[idx] = sm[threadIdx.x - 1] + sm[threadIdx.x] + sm[threadIdx.x + 1];
            }
        }
        else{// Boundary conditions.
            op[idx] = ip[idx];
        }
    }
}

std::vector<float> diff(std::vector<float> const &ip_vec)
{
    // There should be a minimum 2 elements in the input vector.
    if (std::size(ip_vec) < 2u) {
        throw std::invalid_argument{"Input vector should have at least 2 elements."};
    }

    auto const num_elems_diff = std::size(ip_vec) - 1u;
    auto diff_vec = std::vector<float>(num_elems_diff);

    auto const ip_vec_size_bytes = ip_vec.size() * sizeof(float);
    auto const diff_vec_size_bytes = num_elems_diff * sizeof(float);

    // Allocate device memory.
    auto ip_vec_dev = static_cast<float *>(nullptr);
    auto diff_vec_dev = static_cast<float *>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&ip_vec_dev), ip_vec_size_bytes),
               "allocation of device buffer for input vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&diff_vec_dev), diff_vec_size_bytes),
               "allocation of device buffer for diff vector");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(ip_vec_dev, ip_vec.data(), ip_vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the input vector to the device");

    // Execute the kernel.
    // The Number of threads launched depended on the number of elements in the input vector.
    auto const exec_params = ExecConfig::getParams(ip_vec.size(), diff_kernel, 0u);
    diff_kernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        ip_vec_dev, diff_vec_dev, static_cast<unsigned >(ip_vec.size()));

    checkErrorKernel("Diff kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(diff_vec.data(),
                          diff_vec_dev,
                          diff_vec_size_bytes,
                          cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(ip_vec_dev);
    cudaFree(diff_vec_dev);

    return diff_vec;
}

std::vector<float> sum3Point(std::vector<float> const &ip_vec,
                             bool use_shared_mem)
{
    // There should be a minimum 3 elements in the input vector.
    if (std::size(ip_vec) < 3u) {
        throw std::invalid_argument{"Input vector should have at least 3 elements."};
    }

    auto const num_elems = std::size(ip_vec);
    auto sum_vec = std::vector<float>(num_elems, 0.0f);

    auto const vec_size_bytes = num_elems * sizeof(float);

    // Allocate device memory.
    auto ip_vec_dev = static_cast<float *>(nullptr);
    auto sum_vec_dev = static_cast<float *>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&ip_vec_dev), vec_size_bytes),
               "allocation of device buffer for input vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&sum_vec_dev), vec_size_bytes),
               "allocation of device buffer for sum vector");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(ip_vec_dev, ip_vec.data(), vec_size_bytes, cudaMemcpyHostToDevice),
               "transfer of data from the input vector to the device");

    // Execute the kernel.
    if (use_shared_mem) {
        auto const ip_tile_size = 32u;
        auto const op_tile_size = ip_tile_size - 2u;
        auto const block_dim = dim3{ip_tile_size, 1u, 1u};
        auto const grid_size = (num_elems + op_tile_size - 1u) / op_tile_size;
        auto const grid_dim = dim3{static_cast<unsigned>(grid_size), 1u, 1u};
        auto const shared_mem_size = ip_tile_size * sizeof(float);

        sum_3point_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            ip_vec_dev, sum_vec_dev,
            static_cast<unsigned >(ip_vec.size()));
    }
    else {
        // The Number of threads launched depended on the number of elements in the input vector.
        auto const exec_params = ExecConfig::getParams(ip_vec.size(), sum_3point_kernel, 0u);
        sum_3point_kernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
            ip_vec_dev, sum_vec_dev, static_cast<unsigned >(ip_vec.size()));
    }

    checkErrorKernel("sum 3-point kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(sum_vec.data(),
                          sum_vec_dev,
                          vec_size_bytes,
                          cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(ip_vec_dev);
    cudaFree(sum_vec_dev);

    return sum_vec;
}

}// Numeric::CUDA namespace.
