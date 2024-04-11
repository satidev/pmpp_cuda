#include <gmock/gmock.h>
#include "conv_2d.cuh"
#include "../../cpp/eigen_utils.h"

TEST(conv2DTest, exceptionIsThrownSinceDataSizeIsZero)
{
    auto const data = Eigen::MatrixXf{};
    auto const filter = Eigen::MatrixXf::Ones(3, 3);
    EXPECT_THROW(Numeric::CUDA::conv2D(data, filter, false), std::invalid_argument);
}

TEST(conv2DTest, exceptionIsThrownSinceFilterSizeIsZero)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    auto const filter = Eigen::MatrixXf{};
    EXPECT_THROW(Numeric::CUDA::conv2D(data, filter, false), std::invalid_argument);
}

TEST(conv2DTest, exceptionIsThrownSinceFilterIsNotSquare)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    auto const filter = Eigen::MatrixXf::Ones(3, 4);
    EXPECT_THROW(Numeric::CUDA::conv2D(data, filter, false), std::invalid_argument);
}

TEST(conv2DTest, exceptionIsThrownSinceFilterSizeIsEven)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    auto const filter = Eigen::MatrixXf::Ones(2, 2);
    EXPECT_THROW(Numeric::CUDA::conv2D(data, filter, false), std::invalid_argument);
}

TEST(conv2DTest, exceptionIsNotThrownSinceFilterSizeIsOdd)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    auto const filter = Eigen::MatrixXf::Ones(3, 3);
    EXPECT_NO_THROW(Numeric::CUDA::conv2D(data, filter, false));
}

TEST(conv2DTest, outputDataSizeIsSameAsInputDataSize)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    auto const filter = Eigen::MatrixXf::Ones(3, 3);
    auto const res = Numeric::CUDA::conv2D(data, filter, false);
    EXPECT_EQ(res.rows(), data.rows());
    EXPECT_EQ(res.cols(), data.cols());

    auto const res_sm = Numeric::CUDA::conv2D(data, filter, true);
    EXPECT_EQ(res_sm.rows(), data.rows());
    EXPECT_EQ(res_sm.cols(), data.cols());
}

TEST(conv2DTest, correctConv2DResultForIdentityKernel)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3) * 34.0f;
    Eigen::MatrixXf filter = Eigen::MatrixXf::Zero(3, 3);
    filter(1, 1) = 1.0f;

    auto const res = Numeric::CUDA::conv2D(data, filter, false);
    EXPECT_THAT(EigenUtils::toVec(res), ::testing::ContainerEq(EigenUtils::toVec<float>(data)));

    auto const res_sm = Numeric::CUDA::conv2D(data, filter, true);
    EXPECT_THAT(EigenUtils::toVec(res_sm), ::testing::ContainerEq(EigenUtils::toVec<float>(data)));
}

TEST(conv2DTest, correctConv2DOutput)
{
    auto const data = Eigen::MatrixXf::Ones(3, 3);
    Eigen::MatrixXf filter = Eigen::MatrixXf::Ones(3, 3);

    auto const res = Numeric::CUDA::conv2D(data, filter, false);
    // Output is computed by matlab conv2 function with the shape 'same'.
    auto const res_exp = std::vector{4.0f, 6.0f, 4.0f, 6.0f, 9.0f, 6.0f, 4.0f, 6.0f, 4.0f};
    EXPECT_THAT(EigenUtils::toVec(res), ::testing::ContainerEq(res_exp));

    auto const res_sm = Numeric::CUDA::conv2D(data, filter, true);
    EXPECT_THAT(EigenUtils::toVec(res_sm), ::testing::ContainerEq(res_exp));
}


TEST(conv2DTest, correctConv2DOutputLargeData)
{
    auto const data = Eigen::MatrixXf::Ones(101, 101);
    Eigen::MatrixXf filter = Eigen::MatrixXf::Zero(3, 3);
    filter(1, 1) = 1.0f;

    auto const res = Numeric::CUDA::conv2D(data, filter, false);
    EXPECT_THAT(EigenUtils::toVec(res), ::testing::ContainerEq(EigenUtils::toVec<float>(data)));

    auto const res_sm = Numeric::CUDA::conv2D(data, filter, true);
    EXPECT_THAT(EigenUtils::toVec(res_sm), ::testing::ContainerEq(EigenUtils::toVec<float>(data)));
}

