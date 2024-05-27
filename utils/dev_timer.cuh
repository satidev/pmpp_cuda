#ifndef DEV_TIMER_CUH
#define DEV_TIMER_CUH

#include "perf.cuh"

// Timer for cuda events and kernels.
class DevTimer
{
public:
    //TODO: Add stream and device info.
    DevTimer();
    ~DevTimer();

    // Start the clock.
    void tic();
    // End the clock and return elapsed time.
    MilliSeconds toc();

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};


#endif //DEV_TIMER_CUH
