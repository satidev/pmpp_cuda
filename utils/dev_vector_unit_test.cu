#include "stream_adaptor.cuh"
#include "dev_vector.cuh"
#include <gmock/gmock.h>

TEST(DevVectorTest, noExceptionThrownDuringConstruction)
{
    ASSERT_NO_THROW((DevVector<float>{1024u}));
}

TEST(DevVectorTest, dataIsNotNullptrAndSizeIsCorrect)
{
    auto dev_vec = DevVector<float>{1024u};
    ASSERT_TRUE(dev_vec.data() != nullptr);
    ASSERT_EQ(dev_vec.size(), 1024u);
}
