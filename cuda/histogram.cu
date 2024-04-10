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

__global__ void hist_kern_privat_sm(unsigned short const *data,
                                    unsigned *hist,
                                    unsigned num_data_elems,
                                    unsigned num_hist_bins)
{
    // Initialize shared memory to zero.
    // Each block has shared memory to store the histogram values.
    extern __shared__ unsigned hist_sm[];
    // The loop essential if the number of threads is less than the number of bins.
    // If the number of threads is greater than the number of bins
    // the following if condition is sufficient.
    // if(threadIdx.x < num_hist_bins) {
    //   hist_sm[threadIdx.x] = 0u;
    // }
    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        hist_sm[bin] = 0u;
    }
    __syncthreads();

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_data_elems) {

        if (data[idx] == 0) {
            atomicAdd(&(hist_sm[0]), 1u);
        }
        else {
            atomicAdd(&(hist_sm[1]), 1u);
        }
    }
    __syncthreads();// Wait till all threads of the block finish their histogram copy update.

    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        auto const bin_value = hist_sm[bin];
        if (bin_value > 0) {
            atomicAdd(&(hist[bin]), bin_value);
        }
    }

}

std::vector<unsigned> histogramPrivateShared(std::vector<bool> const &data_host)
{
    if (std::empty(data_host)) {
        return std::vector<unsigned>{};
    }
    // Convert input data to unsigned shorts.
    auto const data_i_host = std::vector<unsigned short>(
        std::begin(data_host), std::end(data_host));

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
    auto const shared_mem_bytes = num_hist_bins * sizeof(unsigned);
    auto const exec_params = ExecConfig::getParams(num_data_elems, hist_kern, shared_mem_bytes);
    hist_kern_privat_sm<<<exec_params.grid_dim, exec_params.block_dim, shared_mem_bytes>>>(
        data_i_dev, hist_dev, static_cast<unsigned>(num_data_elems),
        static_cast<unsigned>(num_hist_bins));

    checkErrorKernel("Histogram kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(hist_host), hist_dev, op_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(data_i_dev);
    cudaFree(hist_dev);

    return hist_host;
}

__global__ void hist_kern_privat_sm_coarse_contig(unsigned short const *data,
                                                  unsigned *hist,
                                                  unsigned num_data_elems,
                                                  unsigned num_hist_bins,
                                                  unsigned coarse_factor)
{
    // Initialize shared memory to zero.
    // Each block has shared memory to store the histogram values.
    extern __shared__ unsigned hist_sm[];
    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        hist_sm[bin] = 0u;
    }
    __syncthreads();

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    // More works per each thread to reduce the number of blocks.
    auto const start = idx * coarse_factor;
    auto const end = min(idx * coarse_factor + coarse_factor, num_data_elems);
    for (auto new_idx = start; new_idx < end; new_idx++) {

        if (data[new_idx] == 0) {
            atomicAdd(&(hist_sm[0]), 1u);
        }
        else {
            atomicAdd(&(hist_sm[1]), 1u);
        }
    }
    __syncthreads();// Wait till all threads of the block finish their histogram copy update.

    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        auto const bin_value = hist_sm[bin];
        if (bin_value > 0) {
            atomicAdd(&(hist[bin]), bin_value);
        }
    }

}

__global__ void hist_kern_privat_sm_coarse_interleaved(unsigned short const *data,
                                                       unsigned *hist,
                                                       unsigned num_data_elems,
                                                       unsigned num_hist_bins)
{
    // Initialize shared memory to zero.
    // Each block has shared memory to store the histogram values.
    extern __shared__ unsigned hist_sm[];
    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        hist_sm[bin] = 0u;
    }
    __syncthreads();

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    // More works per each thread to reduce the number of blocks.
    auto const start = idx;
    auto const end = num_data_elems;
    auto const total_num_threads = gridDim.x * blockDim.x;
    for (auto new_idx = start; new_idx < end; new_idx += total_num_threads) {

        if (data[new_idx] == 0) {
            atomicAdd(&(hist_sm[0]), 1u);
        }
        else {
            atomicAdd(&(hist_sm[1]), 1u);
        }
    }
    __syncthreads();// Wait till all threads of the block finish their histogram copy update.

    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        auto const bin_value = hist_sm[bin];
        if (bin_value > 0) {
            atomicAdd(&(hist[bin]), bin_value);
        }
    }

}

std::vector<unsigned> histogramPrivateSharedCoarse(std::vector<bool> const &data_host,
                                                   CoarseningStrategy strategy)
{
    if (std::empty(data_host)) {
        return std::vector<unsigned>{};
    }
    // Convert input data to unsigned shorts.
    auto const data_i_host = std::vector<unsigned short>(
        std::begin(data_host), std::end(data_host));

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
    auto const shared_mem_bytes = num_hist_bins * sizeof(unsigned);
    auto const exec_params = ExecConfig::getParams(num_data_elems, hist_kern, shared_mem_bytes);

    auto const block_size = exec_params.block_dim.x;
    auto const coarse_factor = 16u;
    // Reduction of the number of blocks by factor coarse factor.
    auto const grid_size = static_cast<unsigned>(std::ceil(static_cast<float>(num_data_elems) /
        static_cast<float>(coarse_factor * block_size)));
    switch (strategy) {
        case CoarseningStrategy::CONTIGUOUS_PARTITIONING:
            hist_kern_privat_sm_coarse_contig<<<grid_size, block_size, shared_mem_bytes>>>(
                data_i_dev, hist_dev, static_cast<unsigned>(num_data_elems),
                static_cast<unsigned>(num_hist_bins),
                coarse_factor);
            break;
        case CoarseningStrategy::INTERLEAVED_PARTITIONING:
            hist_kern_privat_sm_coarse_interleaved<<<grid_size, block_size, shared_mem_bytes>>>(
                data_i_dev, hist_dev, static_cast<unsigned>(num_data_elems),
                static_cast<unsigned>(num_hist_bins));
            break;
    }
    checkErrorKernel("Histogram kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(hist_host), hist_dev, op_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(data_i_dev);
    cudaFree(hist_dev);

    return hist_host;
}

__global__ void hist_kern_privat_sm_coarse_interleaved_aggr(unsigned short const *data,
                                                            unsigned *hist,
                                                            unsigned num_data_elems,
                                                            unsigned num_hist_bins)
{
    // Initialize shared memory to zero.
    // Each block has shared memory to store the histogram values.
    extern __shared__ unsigned hist_sm[];
    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        hist_sm[bin] = 0u;
    }
    __syncthreads();

    auto accumulator = 0u;// To store the number of updates aggregated so far.
    auto prev_bin_idx = -1;// Index of the bin that was encounterd last time and aggregated by the accumulator.

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    // More works per each thread to reduce the number of blocks.
    auto const start = idx;
    auto const end = num_data_elems;
    auto const total_num_threads = gridDim.x * blockDim.x;
    for (auto new_idx = start; new_idx < end; new_idx += total_num_threads) {
        auto const bin = data[new_idx];
        if(bin == prev_bin_idx) {
            accumulator++;
        }
        else {
            if(accumulator > 0) {
                atomicAdd(&(hist_sm[prev_bin_idx]), accumulator);
            }
            accumulator = 1u;
            prev_bin_idx = bin;// Just one element behind.
        }

    }

    if(accumulator > 0) {
        atomicAdd(&(hist_sm[prev_bin_idx]), accumulator);
    }

    __syncthreads();// Wait till all threads of the block finish their histogram copy update.

    for (auto bin = threadIdx.x; bin < num_hist_bins; bin += blockDim.x) {
        auto const bin_value = hist_sm[bin];
        if (bin_value > 0) {
            atomicAdd(&(hist[bin]), bin_value);
        }
    }

}

std::vector<unsigned> histogramPrivateSharedCoarseAggr(std::vector<bool> const &data_host)
{
    if (std::empty(data_host)) {
        return std::vector<unsigned>{};
    }
    // Convert input data to unsigned shorts.
    auto const data_i_host = std::vector<unsigned short>(
        std::begin(data_host), std::end(data_host));

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
    auto const shared_mem_bytes = num_hist_bins * sizeof(unsigned);
    auto const exec_params = ExecConfig::getParams(num_data_elems, hist_kern, shared_mem_bytes);

    auto const block_size = exec_params.block_dim.x;
    auto const coarse_factor = 16u;
    // Reduction of the number of blocks by factor coarse factor.
    auto const grid_size = static_cast<unsigned>(std::ceil(static_cast<float>(num_data_elems) /
        static_cast<float>(coarse_factor * block_size)));
    hist_kern_privat_sm_coarse_interleaved_aggr<<<grid_size, block_size, shared_mem_bytes>>>(
        data_i_dev, hist_dev, static_cast<unsigned>(num_data_elems),
        static_cast<unsigned>(num_hist_bins));
    checkErrorKernel("Histogram kernel", true);

    // Transfer result data from the device to host.
    checkError(cudaMemcpy(std::data(hist_host), hist_dev, op_size_bytes, cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(data_i_dev);
    cudaFree(hist_dev);

    return hist_host;
}

}// Numeric::CUDA namespace.
