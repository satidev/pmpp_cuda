#ifndef DEV_TIMER_CUH
#define DEV_TIMER_CUH

#include "perf.cuh"

// Timer for cuda events and kernels.
class DevTimer
{
public:
    DevTimer();
    ~DevTimer();

    // Start the clock.
    void tic(cudaStream_t stream = 0);
    // End the clock and return elapsed time.
    MilliSeconds toc();

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_ = 0;
};

#endif // DEV_TIMER_CUH
