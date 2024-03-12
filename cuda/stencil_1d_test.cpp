#include <gmock/gmock.h>
#include "stencil_1d.cuh"

TEST(cudaStencil1DTest, returnEmptyVecForEmptyInputVector)
{
    ASSERT_TRUE(std::empty(Numeric::CUDA::diff(std::vector<float>{})));
}

TEST(cudaStencil1DTest, outputSizeIsInputSizeMinusOne)
{
    auto const ip_vec = std::vector{1.0f, 2.0f, 3.0f};
    ASSERT_THAT(std::size(Numeric::CUDA::diff(ip_vec)), ip_vec.size() - 1u);
}

TEST(cudaStencil1DTest, returnAllOnes)
{
    auto const ip_vec = std::vector{-1.0f, 2.0f, 3.0f};
    auto const op_exp = std::vector{3.0f, 1.0f};
    ASSERT_THAT(Numeric::CUDA::diff(ip_vec), op_exp);
}