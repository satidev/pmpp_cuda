#include <gmock/gmock.h>
#include "conv.cuh"

TEST(conv1DTest, exceptionIsThrownSinceDataSizeIsZero)
{
    auto const data = std::vector<float>{};
    auto const filter = std::vector<float>{1.0f, 2.0f, 3.0f};
    EXPECT_THROW(Numeric::CUDA::conv1D(data, filter), std::invalid_argument);
}

TEST(conv1DTest, exceptionIsThrownSinceFilterSizeIsZero)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>{};
    EXPECT_THROW(Numeric::CUDA::conv1D(data, filter), std::invalid_argument);
}

TEST(conv1DTest, exceptionThrowSinceFilterSizeIsEven)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>{1.0f, 2.0f};
    EXPECT_THROW(Numeric::CUDA::conv1D(data, filter), std::invalid_argument);
}

TEST(conv1DTest, exceptionIsNotThrownSinceFilterSizeIsOdd)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>{1.0f, 2.0f, 3.0f};
    EXPECT_NO_THROW(Numeric::CUDA::conv1D(data, filter));
}

TEST(conv1DTest, exceptionIsThrownSinceFilterSizeExceedsMaxFilterSizeWhenConstMemIsUsed)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>(1024u + 1u, 1.0f);
    EXPECT_THROW(Numeric::CUDA::conv1D(data, filter, true), std::invalid_argument);
}

TEST(conv1DTest, exceptionIsNotThrownForSharedMem)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>(3u, 1.0f);
    EXPECT_NO_THROW(Numeric::CUDA::conv1D(data, filter, true, true));
}

TEST(conv1DTest, inputAndOutputDataSizeIsSame)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto const filter = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto const res = Numeric::CUDA::conv1D(data, filter);
    EXPECT_EQ(res.size(), data.size());
}

TEST(conv1DTest, correctConv1DResultForIdentityKernel)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto const filter = std::vector<float>{0.0f, 1.0f, 0.0f};
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false, false), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true, false), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false, true), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true, true), ::testing::ContainerEq(data));
}

TEST(conv1DTest, correctConv1DResultForSymmetricFilterKernel)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto const filter = std::vector<float>{1.0f, 2.0f, 1.0f};
    auto const expected = std::vector<float>{4.0f, 8.0f, 12.0f, 16.0f, 14.0f};

    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false, true), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true, true), ::testing::ContainerEq(expected));
}

TEST(conv1DTest, correctConv1DResultForNonSymmFilterKernel)
{
    auto const data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto const filter = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto const expected = std::vector<float>{4.0f, 10.0f, 16.0f, 22.0f, 22.0f};

    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false, true), ::testing::ContainerEq(expected));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true, true), ::testing::ContainerEq(expected));
}


TEST(conv1DTest, correctConv1DResultForIdentityKernelLargeArray)
{
    auto const data = std::vector<float>(2048u, 1.0f);
    auto const filter = std::vector<float>{0.0f, 1.0f, 0.0f};

    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, false, true), ::testing::ContainerEq(data));
    EXPECT_THAT(Numeric::CUDA::conv1D(data, filter, true, true), ::testing::ContainerEq(data));
}