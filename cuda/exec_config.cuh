#ifndef EXEC_CONFIG_CUH
#define EXEC_CONFIG_CUH

#include "dev_config.cuh"
#include <array>
#include <iostream>

// Computes the execution configuration parameters for kernel launch taking account
// of the device properties and the problem size.
// Objective is to maximize the occupancy of the device.
// TODO: Consider register per thread and available shared memory.
struct ExecConfigParams
{
    dim3 grid_dim;
    dim3 block_dim;
};

class ExecConfig
{
public:
    template<class KernFunc>
    static ExecConfigParams getParams(unsigned num_elems,
                                      KernFunc name,
                                      size_t shared_mem_size_bytes = 0);
    template<class KernFunc>
    static float occupancyTheory(DeviceProperties const &dev_prop,
                                 KernFunc name,
                                 ExecConfigParams const &exe_params,
                                 size_t shared_mem_size_bytes);
private:
    template<class KernFunc>
    static unsigned potentialBlockSize(KernFunc name, size_t shared_mem_size_bytes);
};

template<class KernFunc>
unsigned ExecConfig::potentialBlockSize(KernFunc name, size_t shared_mem_size_bytes)
{
    auto min_grid_size = 0, block_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       name, shared_mem_size_bytes, 0);
    return static_cast<unsigned>(block_size);
}
template<class KernFunc>
ExecConfigParams ExecConfig::getParams(unsigned num_elems,
                                       KernFunc name,
                                       size_t shared_mem_size_bytes)
{
    auto const block_size = potentialBlockSize(name, shared_mem_size_bytes);
    auto const num_blocks = (num_elems + block_size - 1u) / block_size;
    return ExecConfigParams{dim3(static_cast<unsigned>(num_blocks), 1u, 1u),
                            dim3(static_cast<unsigned>(block_size), 1u, 1u)};
}
template<class KernFunc>
float ExecConfig::occupancyTheory(const DeviceProperties &dev_prop,
                                  KernFunc name,
                                  ExecConfigParams const &exe_params,
                                  size_t shared_mem_size_bytes)
{
    auto const num_threads_per_block = exe_params.block_dim.x *
        exe_params.block_dim.y * exe_params.block_dim.z;
    auto max_num_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_num_active_blocks,
                                                  name, num_threads_per_block,
                                                  shared_mem_size_bytes);
    auto const max_active_warps = static_cast<float>(
        max_num_active_blocks * num_threads_per_block) / static_cast<float>(dev_prop.warp_size);
    auto const num_warps_per_sm = static_cast<float>(dev_prop.max_threads_per_sm) /
        static_cast<float>(dev_prop.warp_size);

    return (max_active_warps / num_warps_per_sm) * 100.0f;
}

#endif //EXEC_CONFIG_CUH
