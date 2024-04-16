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
    public ::testing::WithParamInterface<std::tuple<unsigned, float, ReductionStrategy>>
{
protected:
    static float testSum(unsigned num_elems, float init_val,
                         ReductionStrategy strategy = ReductionStrategy::SIMPLE)
    {
        if (num_elems == 0u) {
            return sumParallel(std::vector<float>{}, strategy);
        }
        else {
            return sumParallel(std::vector<float>(num_elems, init_val), strategy);
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
    auto [num_elems, init_val, strategy] = GetParam();
    ASSERT_EQ(testSum(num_elems, init_val, strategy), num_elems * init_val);
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
                             std::make_tuple(0u, 1002.0, ReductionStrategy::SIMPLE)
                             ,std::make_tuple(32u, 2.0, ReductionStrategy::NAIVE)
                             ,std::make_tuple(32u, 2.0, ReductionStrategy::SIMPLE)
                             ,std::make_tuple(32u, 1.0, ReductionStrategy::SIMPLE)
                             ,std::make_tuple(512u, 1.0, ReductionStrategy::SIMPLE)
                             ,std::make_tuple(128u, 1.0, ReductionStrategy::SIMPLE_MIN_DIV)
                             ,std::make_tuple(512u, 1.0, ReductionStrategy::SIMPLE_MIN_DIV_SHARED)
                             ,std::make_tuple(128u, 1.0, ReductionStrategy::SIMPLE_MIN_DIV_SHARED_MULT_BLOCKS)
                             ,std::make_tuple(512u, 1.0, ReductionStrategy::SIMPLE_MIN_DIV_SHARED_MULT_BLOCKS_COARSE)
                         )
);


TEST(sumTest, throwsExceptionsForVectorExceedsTwoTimesMaxThreadsPerBlock)
{
    auto const data = std::vector(2 * 1024 + 1, 1.0f);
    ASSERT_THROW(sumParallel(data, ReductionStrategy::SIMPLE), std::invalid_argument);
}
