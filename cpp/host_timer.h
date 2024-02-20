#ifndef HOST_TIMER_H
#define HOST_TIMER_H

#include <chrono>

class HostTimer
{
public:
    void tic();
    float toc();

private:
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> end;
};

#endif //HOST_TIMER_H
