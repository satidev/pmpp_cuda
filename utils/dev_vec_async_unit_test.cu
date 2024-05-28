#include <gmock/gmock.h>
#include "stream_adaptor.cuh"
#include "dev_vector_async.cuh"

TEST(MallocFreeAsyncTest, noThrowOnAllocDealloc)
{
    auto stream = std::make_shared<StreamAdaptor>();
    auto dev_ptr = static_cast<float *>(nullptr);
    ASSERT_NO_THROW(cudaMallocAsync(
        reinterpret_cast<void **>(&dev_ptr), 1024 * sizeof(float), stream->getStream()));
    ASSERT_NO_THROW(cudaFreeAsync(dev_ptr, stream->getStream()));
}

TEST(DevVectorAsyncTest, noExceptionThrownDuringConstruction)
{
    auto stream = StreamAdaptor{};
    ASSERT_NO_THROW((DevVectorAsync<float>{stream, 1024u}));
}

TEST(DevVectorAsyncTest, dataIsNotNullptrAndSizeIsCorrect)
{
    auto stream = StreamAdaptor{};
    auto dev_vec = DevVectorAsync<float>{stream, 1024u};
    ASSERT_TRUE(dev_vec.data() != nullptr);
    ASSERT_EQ(dev_vec.size(), 1024u);
}


