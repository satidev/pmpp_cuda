#include "conv.cuh"
#include "exec_config.cuh"
#include <stdexcept>

namespace Numeric::CUDA
{
__global__ void conv_kern_1d(float const *data,
                             float const *filter,
                             float *res,
                             unsigned num_elems,
                             unsigned filter_radius);

__global__ void conv_kern_1d(float const *data,
                             float const *filter,
                             float *res,
                             unsigned num_elems,
                             unsigned filter_radius)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        auto const filter_size = 2u * filter_radius + 1u;

        // Element by element multiplication and accumulation.
        auto sum = 0.0f;
        for (auto i = 0u; i < filter_size; ++i) {
            auto const data_idx = idx + (filter_radius - i);
            if ((data_idx >= 0u) && (data_idx < num_elems)) {
                sum += data[data_idx] * filter[i];
            }
        }
        res[idx] = sum;
    }
}
// Constant memory.
auto constexpr FILT_SIZE_CONST_MEM = 1024u;
__constant__ float FILTER_CONST[FILT_SIZE_CONST_MEM];

__global__ void conv_kern_1d_const_mem(float const *data,
                                       float *res,
                                       unsigned num_elems,
                                       unsigned filter_radius)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        auto const filter_size = 2u * filter_radius + 1u;

        // Element by element multiplication and accumulation.
        auto sum = 0.0f;
        for (auto i = 0u; i < filter_size; ++i) {
            auto const data_idx = idx + (filter_radius - i);
            if ((data_idx >= 0u) && (data_idx < num_elems)) {
                sum += data[data_idx] * FILTER_CONST[i];
            }
        }
        res[idx] = sum;
    }
}

std::vector<float> conv1D(std::vector<float> const &data,
                          std::vector<float> const &filter,
                          bool use_const_mem)
{
    auto const num_data_elems = data.size();
    // Check input data vector is empty.
    if (num_data_elems == 0u) {
        throw std::invalid_argument{"Data vector is empty"};
    }
    // Check filter vector is empty.
    if (filter.empty()) {
        throw std::invalid_argument{"Filter vector is empty"};
    }
    // Check the number of elements is odd.
    if (filter.size() % 2u == 0u) {
        throw std::invalid_argument{"Filter size should be odd"};
    }

    auto const filter_radius = filter.size() / 2u;
    auto res = std::vector<float>(num_data_elems);
    auto const data_size_bytes = num_data_elems * sizeof(float);
    auto const filter_size_bytes = filter.size() * sizeof(float);

    // Check filter size in bytes exceeds the constant memory size.
    if (use_const_mem && (filter.size() > FILT_SIZE_CONST_MEM)) {
        throw std::invalid_argument{
            "The number of elements in the filter exceeds the constant memory size"};
    }

    // Allocate device memory.
    auto data_dev = static_cast<float *>(nullptr);
    auto filter_dev = static_cast<float *>(nullptr);
    auto res_dev = static_cast<float *>(nullptr);

    cudaMalloc(reinterpret_cast<void **>(&data_dev), data_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&filter_dev), filter_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&res_dev), data_size_bytes);

    // Transfer data from the host to the device.
    cudaMemcpy(data_dev, data.data(), data_size_bytes, cudaMemcpyHostToDevice);
    if(use_const_mem){
        cudaMemcpyToSymbol(FILTER_CONST, filter.data(), filter_size_bytes);
    }
    else{
        cudaMemcpy(filter_dev, filter.data(), filter_size_bytes, cudaMemcpyHostToDevice);
    }

    // Execute the kernel.
    auto const exec_params = ExecConfig::getParams(num_data_elems, conv_kern_1d, 0u);
    if(use_const_mem){
        conv_kern_1d_const_mem<<<exec_params.grid_dim, exec_params.block_dim>>>(
            data_dev, res_dev, static_cast<unsigned>(num_data_elems),
            static_cast<unsigned>(filter_radius));
    }
    else{
        conv_kern_1d<<<exec_params.grid_dim, exec_params.block_dim>>>(
            data_dev, filter_dev, res_dev, static_cast<unsigned>(num_data_elems),
            static_cast<unsigned>(filter_radius));
    }

    // Transfer result data from the device to host.
    cudaMemcpy(res.data(), res_dev, data_size_bytes, cudaMemcpyDeviceToHost);

    // Cleanup device memory.
    cudaFree(data_dev);
    cudaFree(filter_dev);
    cudaFree(res_dev);

    return res;
}

} // Numeric::CUDA namespace.



