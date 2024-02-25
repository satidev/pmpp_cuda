#include <iostream>
#include "../cuda/conv.cuh"
#include "cuda/dev_config.cuh"
#include "cuda/exec_config.cuh"


namespace Numeric::CUDA
{
__global__ void conv_kern_1d(float const *data,
                             float const *filter,
                             float *res,
                             unsigned num_elems,
                             unsigned filter_radius);
}

int main()
{
    try{
        auto constexpr num_elems = 61440u;
        std::cout << "Number of elements:: " << num_elems << std::endl;
        auto const data = std::vector<float>(num_elems, 1.0f);
        auto const filter = std::vector<float>(101, 1.0f);
        auto const res = Numeric::CUDA::conv1D(data, filter, true);

        auto const&dev_config = DeviceConfigSingleton::getInstance();

        // Print occupancy.
        auto const& dev_prop = dev_config.getDevProps(0);
        auto const exec_params = ExecConfig::getParams(num_elems, Numeric::CUDA::conv_kern_1d, 0u);
        auto const occ = ExecConfig::occupancyTheory(dev_prop,
                                                     Numeric::CUDA::conv_kern_1d,
                                                     exec_params,
                                                     0u);
        std::cout << "Occupancy:: " << occ << std::endl;
        std::cout << std::sqrt(24.0) << std::endl;
        // Print number of floating points numbers can be stored in the constant memory.
        std::cout << "Number of floating points numbers can be stored in the constant memory:: " << dev_prop.constant_mem_size / sizeof(float) << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
