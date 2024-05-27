#ifndef STREAM_ADAPTOR_CUH
#define STREAM_ADAPTOR_CUH

#include "check_error.cuh"

class StreamAdaptor
{
private:
    cudaStream_t stream;

public:
    StreamAdaptor()
    {
        checkError(cudaStreamCreate(&stream), "creating stream");
    }

    ~StreamAdaptor()
    {
        checkError(cudaStreamDestroy(stream), "destroying stream");
    }

    cudaStream_t getStream() const
    {
        return stream;
    }
};

#endif //STREAM_ADAPTOR_CUH

