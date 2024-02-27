#include <iostream>
#include "../cuda/conv.cuh"
#include "../cuda/vec_add.cuh"
#include "cuda/dev_config.cuh"
#include "cuda/exec_config.cuh"
#include "../cpp/host_timer.h"

int main()
{
    try {
        auto constexpr num_elems = 1024u*1024u*2u;
        auto const vec_1 = std::vector(num_elems, 1.0f);
        auto const vec_2 = std::vector(num_elems, 2.0f);
        auto const res = Numeric::CUDA::vecAdd(vec_1, vec_2, true);
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
