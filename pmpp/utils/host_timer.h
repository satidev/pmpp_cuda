#ifndef HOST_TIMER_H
#define HOST_TIMER_H

#include <chrono>

class HostTimer
{
public:
    void tic();
    float toc();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
};

#endif //HOST_TIMER_H
