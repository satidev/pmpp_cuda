#include "stencil_1d.cuh"
#include "exec_config.cuh"
#include "check_error.cuh"

namespace Numeric::CUDA
{

__global__ void diff_kernel(float const *ip, float *op,
                            unsigned num_elems_ip)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= 1 && idx < num_elems_ip){
        op[idx - 1] = ip[idx] - ip[idx - 1];
    }
}

std::vector<float> diff(std::vector<float> const &ip_vec)
{
    if(std::empty(ip_vec)){
        return std::vector<float>{};
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
    auto const exec_params = ExecConfig::getParams(ip_vec.size(), diff_kernel, 0u);
    diff_kernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        ip_vec_dev, diff_vec_dev, static_cast<unsigned >(ip_vec.size()));

    checkErrorKernel("Diff kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(diff_vec.data(), diff_vec_dev, diff_vec_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(ip_vec_dev);
    cudaFree(diff_vec_dev);

    return diff_vec;
}

}// Numeric::CUDA namespace.
