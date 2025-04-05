#include "dev_timer.cuh"
#include "check_error.cuh"

DevTimer::DevTimer()
{
    checkError(cudaEventCreate(&start_), "creation of start event of timer");
    checkError(cudaEventCreate(&stop_), "creation of stop event of timer");
}

DevTimer::~DevTimer()
{
    checkError(cudaEventDestroy(start_), "destruction of start event of timer");
    checkError(cudaEventDestroy(stop_), "destruction of stop event of timer");
}
void DevTimer::tic(cudaStream_t stream)
{
    stream_ = stream;
    checkError(cudaEventRecord(start_, stream_), "cudaEventRecord");
}

MilliSeconds DevTimer::toc()
{
    checkError(cudaEventRecord(stop_, stream_), "cudaEventRecord for timer stop function");
    checkError(cudaEventSynchronize(stop_), "cudaEventSynchronize for timer stop function");

    auto elapsed_time_ms = 0.0f;
    checkError(cudaEventElapsedTime(&elapsed_time_ms, start_, stop_),
               "timer's elapsed time computation");
    return MilliSeconds{elapsed_time_ms};
}

