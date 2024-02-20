#include <iostream>
#include "../cuda/vec_add.cuh"

int main()
{
    try{
        auto constexpr num_elems = 512u*512*512u;
        std::cout << "Number of elements:: " << num_elems << std::endl;
        auto const vec_1 = std::vector<float>(num_elems, 1.0f);
        auto const vec_2 = std::vector<float>(num_elems, 2.0f);
        auto const res = Numeric::CUDA::vecAdd(vec_1, vec_2);
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
