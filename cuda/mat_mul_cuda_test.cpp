#include <gmock/gmock.h>
#include "mat_mul.cuh"

TEST(cudaMatMulTest, noThrowsDueToSquareMat)
{
    auto const a = Eigen::Matrix2f();
    ASSERT_NO_THROW(Numeric::CUDA::matMul(a, a));
}


TEST(cudaMatMulTest, throwsDueToNonSquareMat)
{
    auto const a = Eigen::Matrix2f();
    auto const b = Eigen::MatrixXf(2, 3);
    ASSERT_THROW(Numeric::CUDA::matMul(a, b), std::invalid_argument);
    ASSERT_THROW(Numeric::CUDA::matMul(b, a), std::invalid_argument);
}

TEST(cudaMatMulTest, noThrowDueToValidSizeForMatrixMult)
{
    auto const a = Eigen::Matrix2f();
    ASSERT_NO_THROW(Numeric::CUDA::matMul(a, a));
}

TEST(cudaMatMulTest, ThrowDueToInvalidSizeForMatrixMult)
{
    auto const a = Eigen::Matrix2f();
    auto const b = Eigen::Matrix3f();
    ASSERT_THROW(Numeric::CUDA::matMul(a, b), std::invalid_argument);
}


TEST(cudaMatMulTest, squareMatMulTest)
{
    auto a = Eigen::Matrix2f();
    a << 1.0f, 1.0f, 1.0f, 1.0f;
    auto b = Eigen::Matrix2f();
    b << 1.0f, 1.0f, 1.0f, 1.0f;

    auto const exp_res = Numeric::CUDA::matMul(a, b);
    auto const res = a * b;
    ASSERT_THAT(EigenUtils::toVec(exp_res), testing::ContainerEq(EigenUtils::toVec<float>(res)));
}

TEST(cudaMatMulTest, largeSquareMatMulTest)
{
    auto a = Eigen::MatrixXf::Constant(313u, 313u, -1.0f);
    auto b = Eigen::MatrixXf::Constant(313u, 313u, 3.0f);
    auto const exp_res = Numeric::CUDA::matMul(a, b);
    auto const res = a * b;
    ASSERT_THAT(EigenUtils::toVec(exp_res), testing::ContainerEq(EigenUtils::toVec<float>(res)));
}
