#include "sum.cuh"
#include <numeric>
#include "../utils/dev_config.cuh"
#include <stdexcept>
#include <iostream>
#include "../utils/check_error.cuh"

namespace Numeric::CUDA
{
float sumSeq(std::vector<float> const &data)
{
    if (std::empty(data)) {
        return 0.0f;
    }
    else {
        return std::accumulate(std::begin(data), std::end(data), 0.0f);
    }
}
// Similar to the sequential addition.
__global__ void sumKernelNaive(float const *data,
                               float *sum,
                               unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        atomicAdd(sum, data[idx]);
    }
}

__global__ void sumParallelSimple(float *data,
                                  float *sum,
                                  unsigned num_elems)
{
    auto const mem_loc = 2 * threadIdx.x;

    for (auto stride = 1u; stride < num_elems; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            data[mem_loc] += data[mem_loc + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *sum = data[0];
    }
}

// Kernel with reduced warp divergence.
__global__ void sumParallelSimpleMinDiv(float *data,
                                        float *sum,
                                        unsigned num_elems)
{
    auto const mem_loc = threadIdx.x;
    for (auto stride = blockDim.x; stride >= 1u; stride /= 2) {
        if (threadIdx.x < stride) {
            data[mem_loc] += data[mem_loc + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *sum = data[0];
    }
}

// Kernel with reduced warp divergence and shared memory.
__global__ void sumParallelSimpleMinDivShared(float const *data,
                                              float *sum,
                                              unsigned num_elems)
{
    // Copy the result of the first iteration to shared memory.
    extern __shared__ float partial_sum[];
    auto const mem_loc = threadIdx.x;
    partial_sum[mem_loc] = data[mem_loc] + data[mem_loc + blockDim.x];

    for (auto stride = blockDim.x/2u; stride >= 1u; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sum[mem_loc] += partial_sum[mem_loc + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *sum = partial_sum[0];
    }
}

float sumParallel(std::vector<float> const &data_host)
{
    if (std::size(data_host) % 32 != 0) {
        throw std::invalid_argument{"Data size should be a multiple of 32 (warp size)\n"};
    }

    if (std::empty(data_host)) {
        return 0.0f;
    }
    else {
        auto const &dev_config = DeviceConfigSingleton::getInstance().getDevProps(0);
        auto const max_num_threads = dev_config.max_threads_per_block;
        auto const max_num_elems_valid = max_num_threads * 2u;
        if (std::size(data_host) > max_num_elems_valid) {
            throw std::invalid_argument{
                "Data size exceeds the two-time maximum number of threads per block.\n"};
        }

        auto const num_elems = std::size(data_host);
        auto const data_size_bytes = num_elems * sizeof(float);

        auto data_dev = static_cast<float *>(nullptr);
        checkError(cudaMalloc(reinterpret_cast<void **>(&data_dev), data_size_bytes),
                   "allocation of device buffer for data");
        checkError(cudaMemcpy(data_dev,
                              std::data(data_host),
                              data_size_bytes,
                              cudaMemcpyHostToDevice),
                   "transfer of data from the host to the device");

        // Currently, only a single block can be launched.
        // It is better to keep the number of threads per block as a multiple of warp size.
        auto const block_size = static_cast<unsigned>(
            std::ceil(static_cast<float>(num_elems) / 64.0f)) * 32u;
        if (block_size > max_num_threads) {
            throw std::invalid_argument{
                "Block size exceeds the maximum number of threads per block.\n"};
        }
        auto const grid_size = 1u;
        auto sum_dev = static_cast<float *>(nullptr);
        cudaMalloc(reinterpret_cast<void **>(&sum_dev), sizeof(float));
        cudaMemset(sum_dev, 0, sizeof(float));

        sumParallelSimpleMinDivShared<<<grid_size, block_size>>>(data_dev, sum_dev, num_elems);
        checkError(cudaGetLastError(), "launch of sumKernel");

        auto sum_host = 0.0f;
        checkError(cudaMemcpy(&sum_host, sum_dev, sizeof(float), cudaMemcpyDeviceToHost),
                   "transfer of sum from the device to the host");

        cudaFree(data_dev);
        cudaFree(sum_dev);

        return sum_host;
    }
}

}// Numeric::CUDA namespace.
