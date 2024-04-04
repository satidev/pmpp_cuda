#include <iostream>
#include "../cuda/conv_1d.cuh"
#include "../cuda/vec_add.cuh"
#include "cuda/dev_config.cuh"
#include "cuda/exec_config.cuh"
#include "../cpp/host_timer.h"
#include "../cuda/stencil_1d.cuh"

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
