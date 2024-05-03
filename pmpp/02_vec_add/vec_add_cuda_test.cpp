#include <gmock/gmock.h>
#include "vec_add.cuh"
#include "../utils/host_timer.h"

TEST(cudaVecAddTest, exceptThrowSizeMismatch)
{
    EXPECT_THROW(PMPP::CUDA::vecAdd(std::vector{1.0f, 3.0f}, std::vector{2.0f}),
                 std::invalid_argument);
}

TEST(cudaVecAddTest, singleElemVecAdd)
{
    ASSERT_THAT(PMPP::CUDA::vecAdd(std::vector{1.0f}, std::vector{2.0f}),
                testing::ElementsAre(3.0f));
}

TEST(cudaVecAddTest, multiElemVecAdd)
{
    ASSERT_THAT(PMPP::CUDA::vecAdd(std::vector{1.0f, 2.0f, -4.0f},
                                   std::vector{2.0f, 100.0f, 3.0f}),
                testing::ElementsAre(3.0f, 102.0f, -1.0f));
}

TEST(cudaVecAddTest, largeVecAdd)
{

    auto const vec_1 = std::vector<float>(10000, 1.0f);
    auto const vec_2 = std::vector<float>(10000, 2.0f);
    auto const res_exp = std::vector<float>(10000, 3.0f);
    ASSERT_THAT(PMPP::CUDA::vecAdd(vec_1, vec_2), testing::ContainerEq(res_exp));
}


TEST(cudaVecAddTest, performanceTest)
{
    std::cout << "Vector addition:: performance test" << std::endl;
    auto constexpr num_elems = 100000000u;
    auto const vec_1 = std::vector<float>(num_elems, 1.0f);
    auto const vec_2 = std::vector<float>(num_elems, 2.0f);

    auto timer = HostTimer{};
    timer.tic();
    auto res_exp = std::vector<float>{};
    res_exp.reserve(num_elems);
    std::transform(std::begin(vec_1), std::end(vec_1), std::begin(vec_2),
                   std::back_inserter(res_exp),
                   [](float a, float b)
                   {
                       return a + b;
                   }
    );

    std::cout << "Vector addition in CPU:: " << timer.toc() << " seconds." << std::endl;

    timer.tic();
    auto const res = PMPP::CUDA::vecAdd(vec_1, vec_2);
    std::cout << "Vector addition in GPU:: " << timer.toc() << " seconds." << std::endl;

    ASSERT_THAT(res, testing::ContainerEq(res_exp));
}