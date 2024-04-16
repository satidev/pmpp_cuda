#include <iostream>
#include "pmpp/07_convolution/conv_1d.cuh"
#include "pmpp/02_vec_add/vec_add.cuh"
#include "pmpp/utils/dev_config.cuh"
#include "pmpp/utils/exec_config.cuh"
#include "pmpp/utils/host_timer.h"
#include "pmpp/08_stencil/stencil_1d.cuh"

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
