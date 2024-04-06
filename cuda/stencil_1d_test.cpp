#include <gmock/gmock.h>
#include "stencil_1d.cuh"

TEST(cuda1DDiffTest, throwsExceptionForInputVectorWithOneElement)
{
    ASSERT_THROW(Numeric::CUDA::diff(std::vector<float>{1.0f}), std::invalid_argument);
}

TEST(cuda1DDiffTest, outputSizeIsInputSizeMinusOne)
{
    auto const ip_vec = std::vector{1.0f, 2.0f, 3.0f};
    ASSERT_THAT(std::size(Numeric::CUDA::diff(ip_vec)), ip_vec.size() - 1u);
}

TEST(cuda1DDiffTest, returnsCorrectOutput)
{
    auto const ip_vec = std::vector{-1.0f, 2.0f, 3.0f};
    auto const op_exp = std::vector{3.0f, 1.0f};
    ASSERT_THAT(Numeric::CUDA::diff(ip_vec), op_exp);
}

TEST(cuda1DSum3PointTest, throwsExceptionForInputVectorWithLessThanThreeElements)
{
    ASSERT_THROW(Numeric::CUDA::sum3Point(std::vector<float>{1.0f, 2.0f}), std::invalid_argument);
}

TEST(cuda1DSum3PointTest, outputSizeIsSameAsInputSize)
{
    auto const ip_vec = std::vector{1.0f, 2.0f, 3.0f};
    ASSERT_THAT(std::size(Numeric::CUDA::sum3Point(ip_vec)), std::size(ip_vec));
}

TEST(cuda1DSum3PointTest, returnsCorrectOutput)
{
    auto const ip_vec = std::vector{-1.0f, 2.0f, 3.0f};
    auto const op_exp = std::vector{-1.0f, 4.0f, 3.0f};
    ASSERT_THAT(Numeric::CUDA::sum3Point(ip_vec), op_exp);
}

TEST(cuda1DSum3PointTest, correctOutputLargeVector)
{
    auto const ip_vec = std::vector(2048, 1.0f);
    auto op_exp = std::vector(2048, 3.0f);
    op_exp.front() = 1.0f;
    op_exp.back() = 1.0f;

    ASSERT_THAT(Numeric::CUDA::sum3Point(ip_vec), op_exp);
}



