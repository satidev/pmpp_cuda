#include "vec_add.cuh"
#include "vec_add_naive.cuh"
#include "vec_add_cublas.cuh"

namespace PMPP
{
void vecAddPerfTest()
{
    std::cout << "Performance test for vector addition: start" << std::endl;
    auto const num_elems = 1 << 24;
    auto const a = std::vector<float>(num_elems, 1.0f);
    auto const b = std::vector<float>(num_elems, 2.0f);

    auto vec = std::vector<std::pair<VectorAdd<float>, std::string>>{};
    vec.push_back(std::make_pair(VectorAdd<float>{std::make_unique<VecAddNaive<float>>()},
                                 "Naive"));
    vec.push_back(std::make_pair(VectorAdd<float>{std::make_unique<VecAddCublas<float>>()},
                                 "CuBlas"));

    for(auto const &[vec_add, desc]: vec)
    {
        auto const res = std::get<1>(vec_add.run(a, b));
        std::cout << desc << ": " << res.duration_ms << " milli seconds." << std::endl;
    }
    std::cout << "Performance test for vector addition: end" << std::endl;
}
}// namespace PMPP.
