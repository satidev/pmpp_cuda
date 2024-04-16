#include "eigen_utils.h"
#include <gmock/gmock.h>
#include "Eigen/Dense"

TEST(EigenUtilsTest, matToVecCopy)
{
    auto mat = Eigen::MatrixXf{2, 2};
    mat << 1.0f, 2.0f, 3.0f, 4.0f;
    auto const exp_op = std::vector{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_THAT(EigenUtils::toVec<float>(mat), testing::ContainerEq(exp_op));
}

TEST(EigenUtilsTest, vecToSquareMatCopyTest)
{
    auto const vec = std::vector{1.0f, 2.0f, 3.0f, 4.0f};
    auto const mat = EigenUtils::toMat<float>(vec, 2u, 2u);
    ASSERT_EQ(mat(0, 0), 1.0f);
    ASSERT_EQ(mat(0, 1), 2.0f);
    ASSERT_EQ(mat(1, 0), 3.0f);
    ASSERT_EQ(mat(1, 1), 4.0f);
}

TEST(EigenUtilsTest, vecToMatCopyTest)
{
    auto const vec = std::vector{1.0f, 2.0f};
    auto const mat = EigenUtils::toMat<float>(vec, 2u, 1u);
    ASSERT_EQ(mat(0, 0), 1.0f);
    ASSERT_EQ(mat(1, 0), 2.0f);
}