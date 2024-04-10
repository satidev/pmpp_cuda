#include <gmock/gmock.h>
#include "histogram.cuh"

TEST(histTest, emptyVectorOutputWhenInputIsEmpty)
{
    auto const empty_vec = std::vector<bool>{};
    ASSERT_TRUE(std::empty(Numeric::CUDA::histogram(empty_vec)));
    ASSERT_TRUE(std::empty(Numeric::CUDA::histogramPrivatization(empty_vec)));
    ASSERT_TRUE(std::empty(Numeric::CUDA::histogramPrivateShared(empty_vec)));
}

TEST(histTest, uniformHistogram)
{
    auto const data = std::vector<bool>{false, true};
    auto const hist_exp = std::vector{1u, 1u};
    ASSERT_THAT(Numeric::CUDA::histogram(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivatization(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivateShared(data), hist_exp);
}

TEST(histTest, skewedHistogram)
{
    auto const data = std::vector<bool>{true, true};
    auto const hist_exp = std::vector{0u, 2u};
    ASSERT_THAT(Numeric::CUDA::histogram(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivatization(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivateShared(data), hist_exp);
}

TEST(histTest, skewedHistogramLargeVector)
{
    auto const data = std::vector<bool>(2048, true);
    auto const hist_exp = std::vector{0u, 2048u};
    ASSERT_THAT(Numeric::CUDA::histogram(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivatization(data), hist_exp);
    ASSERT_THAT(Numeric::CUDA::histogramPrivateShared(data), hist_exp);
}

