#include <iostream>
#include "cuda/07_convolution/conv_1d.cuh"
#include "../cuda/vec_add.cuh"
#include "cuda/utils/dev_config.cuh"
#include "cuda/utils/exec_config.cuh"
#include "../cpp/host_timer.h"
#include "cuda/08_stencil/stencil_1d.cuh"

int main()
{
    try {
        auto const vec = std::vector<float>{1.0f, 2.0f, 3.0f};
        auto const diff_vec = Numeric::CUDA::diff(vec);
        for(auto const& elem:diff_vec)
            std::cout << elem << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
