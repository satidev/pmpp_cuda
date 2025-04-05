#include "vec_add.cuh"
#include "vec_add_naive.cuh"
#include "vec_add_cublas.cuh"
#include <gmock/gmock.h>

using namespace PMPP;
TEST(vecAddTest, throwExcepForDiffSize)
{
    auto const a = std::vector{1.0f, 2.0f, 3.0f};
    auto const b = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const vec_add = VectorAdd<float>{std::make_unique<VecAddNaive<float>>()};
    ASSERT_THROW(vec_add.run(a, b), std::invalid_argument);
}

TEST(vecAddTest, validRes)
{
    auto const a = std::vector{1.0f, 2.0f, 3.0f};
    auto const b = std::vector{1.0f, 2.0f, 3.0f};
    auto vec = std::vector<VectorAdd<float>>{};
    vec.push_back(VectorAdd<float>{std::make_unique<VecAddNaive<float>>()});
    vec.push_back(VectorAdd<float>{std::make_unique<VecAddCublas<float>>()});

    for (auto const &vec_add : vec)
    {
        auto const res = std::get<0>(vec_add.run(a, b));
        ASSERT_THAT(res, testing::ElementsAre(2.0f, 4.0f, 6.0f));
    }
}
