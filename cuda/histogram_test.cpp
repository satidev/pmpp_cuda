#include <gmock/gmock.h>
#include "histogram.cuh"

using namespace Numeric::CUDA;

TEST(histTest, emptyVectorOutputWhenInputIsEmpty)
{
    auto const empty_vec = std::vector<bool>{};
    ASSERT_TRUE(std::empty(histogram(empty_vec)));
    ASSERT_TRUE(std::empty(histogramPrivatization(empty_vec)));
    ASSERT_TRUE(std::empty(histogramPrivateShared(empty_vec)));
}

TEST(histTest, uniformHistogram)
{
    auto const data = std::vector<bool>{false, true};
    auto const hist_exp = std::vector{1u, 1u};
    ASSERT_THAT(histogram(data), hist_exp);
    ASSERT_THAT(histogramPrivatization(data), hist_exp);
    ASSERT_THAT(histogramPrivateShared(data), hist_exp);
}

TEST(histTest, skewedHistogram)
{
    auto const data = std::vector<bool>{true, true};
    auto const hist_exp = std::vector{0u, 2u};
    ASSERT_THAT(histogram(data), hist_exp);
    ASSERT_THAT(histogramPrivatization(data), hist_exp);
    ASSERT_THAT(histogramPrivateShared(data), hist_exp);
}

TEST(histTest, skewedHistogramLargeVector)
{
    auto const data = std::vector<bool>(2048, true);
    auto const hist_exp = std::vector{0u, 2048u};
    ASSERT_THAT(histogram(data), hist_exp);
    ASSERT_THAT(histogramPrivatization(data), hist_exp);
    ASSERT_THAT(histogramPrivateShared(data), hist_exp);
    ASSERT_THAT(histogramPrivateSharedCoarse(data, CoarseningStrategy::CONTIGUOUS_PARTITIONING), hist_exp);
}

