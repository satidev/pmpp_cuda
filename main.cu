#include <iostream>
#include "../cuda/vec_add.cuh"
#include "cuda/dev_config.cuh"
#include "cuda/exec_config.cuh"


namespace Numeric::CUDA
{
__global__ void vec_add_kern(float const *first,
                             float const *sec,
                             float *res,
                             unsigned num_elems);
}

int main()
{
    try{
        auto constexpr num_elems = 61440u;
        std::cout << "Number of elements:: " << num_elems << std::endl;
        auto const vec_1 = std::vector<float>(num_elems, 1.0f);
        auto const vec_2 = std::vector<float>(num_elems, 2.0f);
        auto const res = Numeric::CUDA::vecAdd(vec_1, vec_2);

        auto const& dev_config = DeviceConfigSingleton::getInstance();

        // Print occupancy.
        auto const& dev_prop = dev_config.getDevProps(0);
        auto const exec_params = ExecConfig::getParams(num_elems, Numeric::CUDA::vec_add_kern, 0u);
        auto const occ = ExecConfig::occupancyTheory(dev_prop,
                                                     Numeric::CUDA::vec_add_kern,
                                                     exec_params,
                                                     0u);
        std::cout << "Occupancy:: " << occ << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
