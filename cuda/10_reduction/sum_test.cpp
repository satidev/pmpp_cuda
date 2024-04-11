#include <gmock/gmock.h>
#include "sum.cuh"

using namespace Numeric::CUDA;

TEST(sumTest, returns0AsSumForEmptyVector)
{
    ASSERT_THAT(sumSeq(std::vector<float>{}), 0.0f);
    ASSERT_THAT(sumParallel(std::vector<float>{}), 0.0f);
}

TEST(sumTest, returns6AsSumSequential)
{
    auto const data = std::vector{1.0f, 2.0f, 3.0f};
    ASSERT_THAT(sumSeq(data), 6.0f);
    ASSERT_THAT(sumParallel(data), 6.0f);
}

TEST(sumTest, throwsExceptionsForVectorExceedsTwoTimesMaxThreadsPerBlock)
{
    auto const data = std::vector(2 * 1024  + 1, 1.0f);
    ASSERT_THROW(sumParallel(data), std::invalid_argument);
}

TEST(sumTest, returns1500AsSumSequential)
{
    auto const data = std::vector<float>(1000, 1.0f);
    ASSERT_THAT(sumSeq(data), 1000.0f);
    ASSERT_THAT(sumParallel(data), 1000.0f);
}

TEST(sumTest, returns1ForSingleElemVector)
{
    auto const data = std::vector<float>{1000.0f};
    ASSERT_THAT(sumSeq(data), 1000.0f);
    ASSERT_THAT(sumParallel(data), 1000.0f);
}

TEST(sumTest, returns1ForDoubleElemVector)
{
    auto const data = std::vector<float>{1000.0f, 1000.0f};
    ASSERT_THAT(sumSeq(data), 2000.0f);
    ASSERT_THAT(sumParallel(data), 2000.0f);
}

TEST(sumTest, returns1ForFourElemVector)
{
    auto const data = std::vector<float>{1000.0f, 1000.0f, 1000.0f, 1000.0f};
    ASSERT_THAT(sumSeq(data), 4000.0f);
    ASSERT_THAT(sumParallel(data), 4000.0f);
}
