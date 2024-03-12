#include <gmock/gmock.h>
#include "stencil_2d.cuh"
#include "../cpp/eigen_utils.h"
#include <stdexcept>

using namespace Numeric::CUDA;
using namespace Eigen;

TEST(cudaStencil2DTest, exceptionIsThrownInvalidMatrixSize)
{
    ASSERT_THROW(sum5PointStencil(MatrixXf(1, 2)), std::invalid_argument);
}

TEST(cudaStencil2DTest, exceptionIsNotThrownValidSize)
{
    ASSERT_NO_THROW(sum5PointStencil(MatrixXf(3, 3)));
}

TEST(cudaStencil2DTest, outputSizeInputSizeMinus2)
{
    auto const res = sum5PointStencil(MatrixXf(7, 8));
    ASSERT_THAT(res.rows(), 5);
    ASSERT_THAT(res.cols(), 6);
}

TEST(cudaStencil2DTest, correctOutput)
{
    auto const ip = Eigen::MatrixXf::Constant(5u, 5u, 1.0f);
    auto const res = sum5PointStencil(ip);
    auto const res_exp = Eigen::MatrixXf::Constant(3u, 3u, 5.0f);
    ASSERT_THAT(EigenUtils::toVec(res), testing::ContainerEq(EigenUtils::toVec<float>(res_exp)));
}