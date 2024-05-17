/*
#include <gmock/gmock.h>
#include "stencil_2d.cuh"
#include <stdexcept>

using namespace PMPP::CUDA;
using namespace Eigen;

TEST(cudaStencil2DTest, exceptionIsThrownInvalidMatrixSize)
{
    ASSERT_THROW(sum5PointStencil(MatrixXf(1, 2)), std::invalid_argument);
}

TEST(cudaStencil2DTest, exceptionIsNotThrownValidSize)
{
    ASSERT_NO_THROW(sum5PointStencil(MatrixXf(3, 3)));
}

TEST(cudaStencil2DTest, outputSizeInputSizeSameAsOutputSize)
{
    auto const res = sum5PointStencil(MatrixXf(7, 8));
    ASSERT_THAT(res.rows(), 7);
    ASSERT_THAT(res.cols(), 8);
}

TEST(cudaStencil2DTest, correctOutput)
{
    auto const num_rows = 1234;
    auto const num_cols = 126;
    auto const ip = Eigen::MatrixXf::Constant(num_rows, num_cols, 1.0f);

    Eigen::MatrixXf res_exp = Eigen::MatrixXf::Constant(num_rows, num_cols, 1.0f);

    for(auto i = 1u; i < num_rows - 1; ++i) {
        for(auto j = 1u; j < num_cols - 1; ++j) {
            res_exp(i, j) = 5.0f;
        }
    }

    auto const res = sum5PointStencil(ip, false);
    ASSERT_THAT(EigenUtils::toVec(res), testing::ContainerEq(EigenUtils::toVec<float>(res_exp)));

    auto const res_sm = sum5PointStencil(ip, true);
    ASSERT_THAT(EigenUtils::toVec(res_sm), testing::ContainerEq(EigenUtils::toVec<float>(res_exp)));
}
*/
