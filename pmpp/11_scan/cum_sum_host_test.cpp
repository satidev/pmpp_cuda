#include "cum_sum_host.h"
#include <gmock/gmock.h>

TEST(cumSumHostTest, emptyVecCumSum)
{
    ASSERT_THAT(PMPP::cumSumHost(std::vector<float>{}), testing::ElementsAre());
}

TEST(cumSumHostTest, singleElemCumSum)
{
    ASSERT_THAT(PMPP::cumSumHost(std::vector{1.0f}), testing::ElementsAre(1.0f));
}

TEST(cumSumHostTest, twoElemCumSum)
{
    ASSERT_THAT(PMPP::cumSumHost(std::vector{1.0f, 2.0f}), testing::ElementsAre(1.0f, 3.0f));
}

TEST(cumSumHostTest, multiElemCumSum)
{
    ASSERT_THAT(PMPP::cumSumHost(std::vector{1.0f, 2.0f, 3.0f, 4.0f}),
                testing::ElementsAre(1.0f, 3.0f, 6.0f, 10.0f));
}
