#include <iostream>
#include "../cuda/conv.cuh"
#include "cuda/dev_config.cuh"
#include "cuda/exec_config.cuh"
#include "../cpp/host_timer.h"

int main()
{
    try {
        auto constexpr num_elems = 61440u;
        std::cout << "Number of elements:: " << num_elems << std::endl;
        auto const data = std::vector<float>(num_elems, 1.0f);
        auto const filter = std::vector<float>(765, 1.0f);
        auto timer = HostTimer{};
        timer.tic();
        auto const res_1 = Numeric::CUDA::conv1D(data, filter, false, false);
        std::cout << "Only global memory:: " << timer.toc() << std::endl;
        timer.tic();
        auto const res_2 = Numeric::CUDA::conv1D(data, filter, false, true);
        std::cout << "Using shared memory:: " << timer.toc() << std::endl;
        timer.tic();
        auto const res_3 = Numeric::CUDA::conv1D(data, filter, true, false);
        std::cout << "Using constant memory:: " << timer.toc() << std::endl;
        timer.tic();
        auto const res_4 = Numeric::CUDA::conv1D(data, filter, true, true);
        std::cout << "Using shared and constant memory:: " << timer.toc() << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
