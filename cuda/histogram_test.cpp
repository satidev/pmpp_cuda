#include <gmock/gmock.h>
#include "histogram.cuh"

TEST(histTest, emptyVectorOutputWhenInputIsEmpty)
{
    auto const empty_vec = std::vector<bool>{};
    ASSERT_TRUE(std::empty(Numeric::CUDA::histogram(empty_vec)));
}

TEST(histTest, uniformHistogram)
{
    auto const data = std::vector<bool>{false, true};
    auto const hist_exp = std::vector{1u, 1u};
    ASSERT_THAT(Numeric::CUDA::histogram(data), hist_exp);
}