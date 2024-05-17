#include "vec_add.cuh"
#include "../utils/exec_config.cuh"
#include <stdexcept>
#include "../utils/check_error.cuh"
#include "../utils/dev_timer.cuh"
#include "../utils/dev_vector.cuh"
#include "cuda_profiler_api.h"
#include "../utils/dev_vector_factory.cuh"
#include "../utils/host_dev_copy.cuh"

namespace PMPP::CUDA
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
    auto first_dev = DevVectorFactory::create(first_host);
    auto sec_dev = DevVectorFactory::create(sec_host);

    // Allocate result vector in the device.
    auto const num_elems{static_cast<unsigned>(std::size(first_host))};
    auto res_dev{DevVector<float>{num_elems}};

    // Execute the kernel.
    auto const exec_params{ExecConfig::getParams(num_elems, vecAddKernel, 0u)};
    auto timer{DevTimer{}};
    timer.tic();

    //cudaProfilerStart();
    vecAddKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        first_dev.data(), sec_dev.data(), res_dev.data(), num_elems);
    checkErrorKernel("vector addition kernel", true);
    //cudaProfilerStop();
    auto time_taken_sec{timer.toc()};

    if (print_kernel_time) {
        std::cout << "Time taken (kernel:vec_add):: " << time_taken_sec * 1000.0 << " milli seconds." << std::endl;
    }

    return HostDevCopy::hostCopy(res_dev);
}

void vecAddExample()
{
    auto const dev_prop = DeviceConfigSingleton::getInstance().getDevProps(0);
    auto const max_num_thread_sm = dev_prop.max_threads_per_sm;
    auto const num_sm = dev_prop.num_sm;

    auto const num_elems = max_num_thread_sm * num_sm * 1000u;
    auto const first_host = std::vector<float>(num_elems, 1.0f);
    auto const sec_host = std::vector<float>(num_elems, 2.0f);
    vecAdd(first_host, sec_host, true);
}

}// Numeric namespace.

