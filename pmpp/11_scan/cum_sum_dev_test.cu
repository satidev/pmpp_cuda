#include <gmock/gmock.h>
#include <random>
#include "cum_sum_host.h"
#include "cum_sum_dev.cuh"

TEST(cumSumDevTest, emptyVecCumSum)
{
    ASSERT_THAT(PMPP::cumSumDev(std::vector<float>{}), testing::ElementsAre());
}

TEST(cumSumDevTest, returnsCorrResultForUniformVec)
{
    auto const data = std::vector(512u, 1.0f);
    auto const res_exp = PMPP::cumSumHost(data);
    ASSERT_THAT(PMPP::cumSumDev(data), testing::ContainerEq(res_exp));
}


