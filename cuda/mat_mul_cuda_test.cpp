#include <gmock/gmock.h>
#include "mat_mul.cuh"
#include <iostream>

TEST(cudaMatMulTest, noThrowDueToValidSizeForMatrixMult)
{
    auto const a = Eigen::MatrixXf(2, 3);
    auto const b = Eigen::MatrixXf(3, 4);
    ASSERT_NO_THROW(Numeric::CUDA::matMul(a, b));
    ASSERT_NO_THROW(Numeric::CUDA::matMul(Eigen::Matrix2f(), Eigen::Matrix2f()));
}

TEST(cudaMatMulTest, ThrowDueToInvalidSizeForMatrixMult)
{
    auto const a = Eigen::Matrix2f();
    auto const b = Eigen::Matrix3f();
    ASSERT_THROW(Numeric::CUDA::matMul(a, b), std::invalid_argument);
}


TEST(cudaMatMulTest, squareMatMulTest)
{
    auto const a = Eigen::MatrixXf::Constant(2u, 2u, 1.0f);
    auto const b = Eigen::MatrixXf::Constant(2u, 2u, 1.0f);

    auto const exp_res = a * b;

    auto const res = Numeric::CUDA::matMul(a, b);
    ASSERT_THAT(EigenUtils::toVec(res), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));

    auto const res_sm = Numeric::CUDA::matMul(a, b, true);
    ASSERT_THAT(EigenUtils::toVec(res_sm), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));
}

TEST(cudaMatMulTest, validResMatrixSizeTest)
{
    auto const a = Eigen::MatrixXf::Constant(112, 234, -1.0f);
    auto const b = Eigen::MatrixXf::Constant(234, 118, 3.0f);

    auto const res = Numeric::CUDA::matMul(a, b);

    ASSERT_EQ(res.rows(), a.rows());
    ASSERT_EQ(res.cols(), b.cols());
}

TEST(cudaMatMulTest, genMatrixMultTest)
{
    auto const a = Eigen::MatrixXf::Constant(115u, 21u, 1.0f);
    auto const b = Eigen::MatrixXf::Constant(21u, 314u, 1.0f);

    auto const exp_res = a * b;

    auto const res = Numeric::CUDA::matMul(a, b);
    ASSERT_THAT(EigenUtils::toVec(res), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));

    auto const res_sm = Numeric::CUDA::matMul(a, b, true);
    ASSERT_THAT(EigenUtils::toVec(res_sm), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));
}

TEST(cudaMatMulTest, largeMatrixMultTest)
{
    auto const a = Eigen::MatrixXf::Constant(3456u, 1245u, 1.0f);
    auto const b = Eigen::MatrixXf::Constant(1245u, 2345u, 1.0f);

    auto const exp_res = a * b;

    auto const res = Numeric::CUDA::matMul(a, b);
    ASSERT_THAT(EigenUtils::toVec(res), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));

    auto const res_sm = Numeric::CUDA::matMul(a, b, true);
    ASSERT_THAT(EigenUtils::toVec(res_sm), testing::ContainerEq(EigenUtils::toVec<float>(exp_res)));
}
