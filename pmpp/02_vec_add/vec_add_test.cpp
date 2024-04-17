#include <gmock/gmock.h>
#include "vec_add.h"

TEST(vecAddTest, exceptThrowSizeMismatch) {
    EXPECT_THROW(Numeric::vecAdd(std::vector{1.0f, 3.0f}, std::vector{2.0f}),
                std::invalid_argument);
}

TEST(vecAddTest, singleElemVecAdd) {
    ASSERT_THAT(Numeric::vecAdd(std::vector{1.0f}, std::vector{2.0f}),
                testing::ElementsAre(3.0f));
}

TEST(vecAddTest, multiElemVecAdd) {
    ASSERT_THAT(Numeric::vecAdd(std::vector{1.0f, 2.0f, -4.0f},
                                std::vector{2.0f, 100.0f, 3.0f}),
                testing::ElementsAre(3.0f, 102.0f, -1.0f));
}

TEST(vecAddTest, largeVecAdd) {

    auto const vec_1 = std::vector<float>(10000, 1.0f);
    auto const vec_2 = std::vector<float>(10000, 2.0f);
    auto const res_exp = std::vector<float>(10000, 3.0f);
    ASSERT_THAT(Numeric::vecAdd(vec_1, vec_2), testing::ContainerEq(res_exp));
}