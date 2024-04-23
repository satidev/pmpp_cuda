#include "vec_add.cuh"
#include "../utils/exec_config.cuh"
#include <stdexcept>
#include "../utils/check_error.cuh"
#include "../utils/timer.cuh"
#include "../utils/dev_vector.cuh"

namespace Numeric::CUDA
{
__global__ void vecAddKernel(float const *first, float const *sec,
                             float *res, unsigned num_elems);

__global__ void vecAddKernel(float const *first, float const *sec,
                             float *res, unsigned num_elems)
{
    auto const idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < num_elems) {
        res[idx] = first[idx] + sec[idx];
    }
}

std::vector<float> vecAdd(std::vector<float> const &first_host,
                          std::vector<float> const &sec_host,
                          bool print_kernel_time)
{
    if (std::size(first_host) != std::size(sec_host)) {
        throw std::invalid_argument{"Vector size should be equal\n"};
    }

    // Allocate device vectors and transfer input data.
    auto first_dev{DevVector{first_host}};
    auto sec_dev{DevVector{sec_host}};

    // Allocate result vector in the device.
    auto const num_elems{static_cast<unsigned>(std::size(first_host))};
    auto res_dev{DevVector<float>{num_elems}};

    // Execute the kernel.
    auto const exec_params{ExecConfig::getParams(num_elems, vecAddKernel, 0u)};
    auto timer{Timer{}};
    timer.tic();

    vecAddKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        first_dev.data(), sec_dev.data(), res_dev.data(), num_elems);
    checkErrorKernel("vector addition kernel", true);

    auto time_taken_sec{timer.toc()};
    if (print_kernel_time) {
        std::cout << "Time taken (kernel:vec_add):: " << time_taken_sec << " seconds." << std::endl;
    }

    return res_dev.hostCopy();
}

}// Numeric namespace.

