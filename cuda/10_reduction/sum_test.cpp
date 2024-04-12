#include <gmock/gmock.h>
#include "sum.cuh"

using namespace Numeric::CUDA;

class SumSeqTestFixture:
    public ::testing::Test,
    public ::testing::WithParamInterface<std::pair<unsigned, float>>
{
protected:
    static float testSum(unsigned num_elems, float init_val)
    {
        if (num_elems == 0u) {
            return sumSeq(std::vector<float>{});
        }
        else {
            return sumSeq(std::vector<float>(num_elems, init_val));
        }
    }
};

class SumParallelTestFixture:
    public ::testing::Test,
    public ::testing::WithParamInterface<std::pair<unsigned, float>>
{
protected:
    static float testSum(unsigned num_elems, float init_val)
    {
        if (num_elems == 0u) {
            return sumParallel(std::vector<float>{});
        }
        else {
            return sumParallel(std::vector<float>(num_elems, init_val));
        }
    }
};

TEST_P(SumSeqTestFixture, checkSumSeq)
{
    auto [num_elems, init_val] = GetParam();
    ASSERT_EQ(testSum(num_elems, init_val), num_elems * init_val);
}

TEST_P(SumParallelTestFixture, checkSumParallel)
{
    auto [num_elems, init_val] = GetParam();
    ASSERT_EQ(testSum(num_elems, init_val), num_elems * init_val);
}

INSTANTIATE_TEST_SUITE_P(SumTest, SumSeqTestFixture,
                         ::testing::Values(
                             std::make_pair(0u, 1002.0),
                             std::make_pair(32u, 2.0),
                             std::make_pair(32u, 1.0)
                         )
);

INSTANTIATE_TEST_SUITE_P(SumTest, SumParallelTestFixture,
                         ::testing::Values(
                             std::make_pair(0u, 1002.0),
                             std::make_pair(32u, 2.0),
                             std::make_pair(32u, 1.0),
                             std::make_pair(128u, 1.0)
                         )
);


TEST(sumTest, throwsExceptionsForVectorExceedsTwoTimesMaxThreadsPerBlock)
{
    auto const data = std::vector(2 * 1024 + 1, 1.0f);
    ASSERT_THROW(sumParallel(data), std::invalid_argument);
}
