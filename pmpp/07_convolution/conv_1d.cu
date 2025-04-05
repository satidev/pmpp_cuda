#include "conv_1d.cuh"
#include "../utils/exec_config.cuh"
#include <stdexcept>

namespace PMPP::CUDA
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

        if (idx < num_elems)
        {

            auto const filter_size = 2u * filter_radius + 1u;
            // Element by element multiplication and accumulation.
            auto sum = 0.0f;
            for (auto filt_idx = 0u; filt_idx < filter_size; ++filt_idx)
            {
                // Flipped data index.
                auto const data_idx = static_cast<int>(idx) - static_cast<int>(filt_idx) +
                                      static_cast<int>(filter_radius);
                if (data_idx >= 0 && data_idx < num_elems)
                { // Check the data is available (ghost
                    // cells).
                    sum += data[data_idx] * filter[filt_idx];
                } // If the data is not available, add zero to the sum.
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
        if (idx < num_elems)
        {
            auto const filter_size = 2u * filter_radius + 1u;
            // Element by element multiplication and accumulation.
            auto sum = 0.0f;
            for (auto filt_idx = 0u; filt_idx < filter_size; ++filt_idx)
            {
                auto const data_idx = static_cast<int>(idx) - static_cast<int>(filt_idx) +
                                      static_cast<int>(filter_radius);
                if (data_idx >= 0 && data_idx < num_elems)
                {
                    sum += data[data_idx] * FILTER_CONST[filt_idx];
                }
            }
            res[idx] = sum;
        }
    }

    // Using shared memory.
    __global__ void conv_kern_1d_sm(float const *data,
                                    float const *filter,
                                    float *res,
                                    unsigned num_elems,
                                    unsigned filter_radius,
                                    unsigned ip_tile_width,
                                    unsigned op_tile_width);

    __global__ void conv_kern_1d_sm(float const *data,
                                    float const *filter,
                                    float *res,
                                    unsigned num_elems,
                                    unsigned filter_radius)
    {
        auto const ip_tile_width = static_cast<int>(blockDim.x);
        // Based on op_tile_width, the grid size is calculated.
        auto const op_tile_width = static_cast<int>(ip_tile_width) -
                                   static_cast<int>(2u * filter_radius);

        auto const idx = static_cast<int>(blockIdx.x * op_tile_width + threadIdx.x) -
                         static_cast<int>(filter_radius);

        // Copy the tile to shared memory.
        extern __shared__ float sm[];
        if (idx >= 0 && idx < static_cast<int>(num_elems))
        {
            sm[threadIdx.x] = data[idx];
        }
        else
        {
            sm[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        auto const tile_idx = static_cast<int>(threadIdx.x) -
                              static_cast<int>(filter_radius);

        if (idx >= 0 && idx < static_cast<int>(num_elems))
        {
            if (tile_idx >= 0 && tile_idx < op_tile_width)
            {
                auto sum = 0.0f;
                auto const filter_size = 2u * filter_radius + 1u;
                for (auto filt_idx = 0u; filt_idx < filter_size; ++filt_idx)
                {
                    // Instead of data, kernel index is flipped.
                    sum += sm[tile_idx + filt_idx] * filter[filter_size - 1 - filt_idx];
                }
                res[idx] = sum;
            }
        }
    }

    __global__ void conv_kern_1d_const_mem_sm(float const *data,
                                              float *res,
                                              unsigned num_elems,
                                              unsigned filter_radius);

    __global__ void conv_kern_1d_const_mem_sm(float const *data,
                                              float *res,
                                              unsigned num_elems,
                                              unsigned filter_radius)
    {
        auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
        // Load the tile to shared memory.
        extern __shared__ float sm[];
        if (idx < num_elems)
        {
            sm[threadIdx.x] = data[idx];
        }
        else
        {
            sm[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (idx < num_elems)
        {
            auto const filter_size = 2u * filter_radius + 1u;

            // Element by element multiplication and accumulation.
            auto sum = 0.0f;
            for (auto filt_idx = 0u; filt_idx < filter_size; ++filt_idx)
            {
                auto const sm_idx = static_cast<int>(threadIdx.x) -
                                    static_cast<int>(filter_radius) + static_cast<int>(filt_idx);
                // Check the data is available in shared memory.
                if (sm_idx >= 0 && sm_idx < static_cast<int>(blockDim.x))
                {
                    // Instead of data, kernel index is flipped.
                    sum += FILTER_CONST[filter_size - 1 - filt_idx] * sm[sm_idx];
                }
                else
                { // If the data is not in shared memory, fetch from global memory.
                    auto const data_idx = static_cast<int>(idx) -
                                          static_cast<int>(filter_radius) + static_cast<int>(filt_idx);
                    if (data_idx >= 0 && data_idx < num_elems)
                    {
                        sum += FILTER_CONST[filter_size - 1 - filt_idx] * data[data_idx];
                    }
                }
            }
            res[idx] = sum;
        }
    }

    std::vector<float> conv1D(std::vector<float> const &data,
                              std::vector<float> const &filter,
                              bool use_const_mem,
                              bool use_shared_mem)
    {
        auto const num_data_elems = data.size();
        // Check input data vector is empty.
        if (num_data_elems == 0u)
        {
            throw std::invalid_argument{"Data vector is empty"};
        }

        // Check filter vector is empty.
        if (filter.empty())
        {
            throw std::invalid_argument{"Filter vector is empty"};
        }

        // Check the number of elements is odd.
        if (filter.size() % 2u == 0u)
        {
            throw std::invalid_argument{"Filter size should be odd"};
        }

        auto const filter_radius = filter.size() / 2u;
        auto res = std::vector<float>(num_data_elems);
        auto const data_size_bytes = num_data_elems * sizeof(float);
        auto const filter_size_bytes = filter.size() * sizeof(float);

        // Check filter size in bytes exceeds the constant memory size.
        if (use_const_mem && (filter.size() > FILT_SIZE_CONST_MEM))
        {
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

        if (use_const_mem)
        {
            cudaMemcpyToSymbol(FILTER_CONST, filter.data(), filter_size_bytes);
        }
        else
        {
            cudaMemcpy(filter_dev, filter.data(), filter_size_bytes, cudaMemcpyHostToDevice);
        }

        // Execute the kernel.
        if (use_shared_mem)
        {
            auto constexpr ip_tile_width = 768u;
            auto const shared_mem_size = ip_tile_width * sizeof(float);
            auto const &dev_config = DeviceConfigSingleton::getInstance();
            auto const shared_mem_per_blk = dev_config.getDevProps(0).max_shared_mem_per_block;
            if (shared_mem_size > shared_mem_per_blk)
            {
                throw std::invalid_argument{"The shared memory size exceeds the device limit"};
            }

            if (use_const_mem)
            {
                auto const block_size = ip_tile_width;
                auto const grid_size = (num_data_elems + ip_tile_width - 1u) / ip_tile_width;
                conv_kern_1d_const_mem_sm<<<grid_size, block_size, shared_mem_size>>>(
                    data_dev, res_dev, static_cast<unsigned>(num_data_elems),
                    static_cast<unsigned>(filter_radius));
            }
            else
            {
                auto const block_size = ip_tile_width;
                auto const op_tile_width = ip_tile_width - 2u * filter_radius;
                auto const grid_size = (num_data_elems + op_tile_width - 1u) / op_tile_width;
                conv_kern_1d_sm<<<grid_size, block_size, shared_mem_size>>>(
                    data_dev, filter_dev, res_dev, static_cast<unsigned>(num_data_elems),
                    static_cast<unsigned>(filter_radius));
            }
        }
        else
        {
            auto const exec_params = ExecConfig::getParams(num_data_elems, conv_kern_1d, 0u);
            if (use_const_mem)
            {
                conv_kern_1d_const_mem<<<exec_params.grid_dim, exec_params.block_dim>>>(
                    data_dev, res_dev, static_cast<unsigned>(num_data_elems),
                    static_cast<unsigned>(filter_radius));
            }
            else
            {
                conv_kern_1d<<<exec_params.grid_dim, exec_params.block_dim>>>(
                    data_dev, filter_dev, res_dev, static_cast<unsigned>(num_data_elems),
                    static_cast<unsigned>(filter_radius));
            }
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
