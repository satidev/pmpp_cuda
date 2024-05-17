#include "host_timer.h"

void HostTimer::tic()
{
    start = std::chrono::high_resolution_clock::now();
}
float HostTimer::toc()
{
    end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>{end - start}.count()/1e9;
}
