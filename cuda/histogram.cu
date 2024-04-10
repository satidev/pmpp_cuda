#include "histogram.cuh"
#include "check_error.cuh"
#include "exec_config.cuh"

namespace Numeric::CUDA
{
__global__ void hist_kern(unsigned short const *data,
                          unsigned *hist,
                          unsigned num_data_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_data_elems) {

        if (data[idx] == 0) {
            atomicAdd(&(hist[0]), 1u);
        }
        else {
            atomicAdd(&(hist[1]), 1u);
        }
    }
}

std::vector<unsigned> histogram(std::vector<bool> const &data_host)
{
    if (std::empty(data_host)) {
        return std::vector<unsigned>{};
    }
    // Convert input data to unsigned shorts.
    auto const
        data_i_host = std::vector<unsigned short>(std::begin(data_host), std::end(data_host));

    auto const num_data_elems = std::size(data_i_host);
    auto const num_hist_bins = 2u;

    // Histogram output in host.
    auto hist_host = std::vector<unsigned>(num_hist_bins);

    auto const ip_size_bytes = num_data_elems * sizeof(unsigned short);
    auto const op_size_bytes = num_hist_bins * sizeof(unsigned);

    // Copy input data to the device.
    auto data_i_dev = static_cast<unsigned short *>(nullptr);
    auto hist_dev = static_cast<unsigned *>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&data_i_dev), ip_size_bytes),
               "allocation of device buffer for input vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&hist_dev), op_size_bytes),
               "allocation of device buffer for histogram output");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(data_i_dev,
                          std::data(data_i_host),
                          ip_size_bytes,
                          cudaMemcpyHostToDevice),
               "transfer of data from the input vector to the device");

    // Initialize histogram buffer to zero.
    checkError(cudaMemset(hist_dev, 0u, op_size_bytes),
               "Initialize histogram buffer");
    // Execute the kernel.
    // The Number of threads launched depended on the number of elements in the input vector.
    auto const exec_params = ExecConfig::getParams(num_data_elems, hist_kern, 0u);
    hist_kern<<<exec_params.grid_dim, exec_params.block_dim>>>(
        data_i_dev, hist_dev, static_cast<unsigned>(num_data_elems));

    checkErrorKernel("Histogram kernel", true);


    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(hist_host), hist_dev, op_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(data_i_dev);
    cudaFree(hist_dev);

    return hist_host;
}

__global__ void hist_kern_privat(unsigned short const *data,
                                 unsigned *hist,
                                 unsigned num_data_elems,
                                 unsigned num_hist_bins)
{

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_data_elems) {

        if (data[idx] == 0) {
            atomicAdd(&(hist[blockIdx.x * num_hist_bins + 0]), 1u);
        }
        else {
            atomicAdd(&(hist[blockIdx.x * num_hist_bins + 1]), 1u);
        }
    }

    if (blockIdx.x > 0) {// Should not add the first block histogram value to itself.
        __syncthreads();// Wait till all threads of the block finish their histogram copy update.
        // The Number of bins can be different from the number of threads per block.
        // Some threads can be responsible for multiple bins, especially
        // block size is larger than the number of bins.
        for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
            auto const bin_value = hist[blockIdx.x * num_hist_bins + bin];
            if (bin_value > 0) {
                atomicAdd(&(hist[bin]), bin_value);
            }
        }
    }

}

__global__ void merge_hist_kern(unsigned *hist,
                                unsigned num_hist_copy,
                                unsigned num_hist_bins)
{
    auto const row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_hist_copy && col < num_hist_bins) {
        auto const hist_idx = row * num_hist_bins + col;
        if (hist_idx >= num_hist_bins) {
            auto const hist_val = hist[hist_idx];
            atomicAdd(&(hist[col]), hist_val);
        }

    }

}

std::vector<unsigned> histogramPrivatization(std::vector<bool> const &data_host)
{
    if (std::empty(data_host)) {
        return std::vector<unsigned>{};
    }
    // Convert input data to unsigned shorts.
    auto const
        data_i_host = std::vector<unsigned short>(std::begin(data_host), std::end(data_host));

    auto const num_data_elems = std::size(data_i_host);
    auto const num_hist_bins = 2u;

    // Histogram output in host.
    auto hist_host = std::vector<unsigned>(num_hist_bins);

    auto const ip_size_bytes = num_data_elems * sizeof(unsigned short);
    auto const op_size_bytes = num_hist_bins * sizeof(unsigned);

    // Copy input data to the device.
    auto data_i_dev = static_cast<unsigned short *>(nullptr);
    auto hist_dev = static_cast<unsigned *>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&data_i_dev), ip_size_bytes),
               "allocation of device buffer for input vector");

    auto const exec_params = ExecConfig::getParams(num_data_elems, hist_kern, 0u);
    auto const mult_hist_bytes = exec_params.grid_dim.x * num_hist_bins * sizeof(unsigned);
    checkError(cudaMalloc(reinterpret_cast<void **>(&hist_dev), mult_hist_bytes),
               "allocation of device buffer for histogram output");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(data_i_dev,
                          std::data(data_i_host),
                          ip_size_bytes,
                          cudaMemcpyHostToDevice),
               "transfer of data from the input vector to the device");

    // Initialize histogram buffer to zero.
    checkError(cudaMemset(hist_dev, 0u, mult_hist_bytes),
               "Initialize histogram buffer");
    // Execute the kernel.
    // The Number of threads launched depended on the number of elements in the input vector.

    hist_kern_privat<<<exec_params.grid_dim, exec_params.block_dim>>>(
        data_i_dev, hist_dev,
        static_cast<unsigned>(num_data_elems),
        static_cast<unsigned>(num_hist_bins));

    checkErrorKernel("Histogram kernel using privatization", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(hist_host), hist_dev, op_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(data_i_dev);
    cudaFree(hist_dev);

    return hist_host;
}

}// Numeric::CUDA namespace.
