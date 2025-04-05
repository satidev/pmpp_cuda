#ifndef PERF_CUH
#define PERF_CUH

#include <string>
#include <vector>
#include <map>
#include <chrono>

using MilliSeconds = std::chrono::duration<float, std::milli>;

struct kernelPerfInfo
{
    MilliSeconds exec_duration;
};
using PerfTestResult = std::map<std::string, std::vector<float>>;

// Returns bandwidth in GB/s.
inline float computeBandwidth(size_t num_bytes, MilliSeconds duration)
{
    auto const num_gb = static_cast<float>(num_bytes) /
                        static_cast<float>(1 << 30);
    return num_gb * 1000.0f / duration.count();
}

#endif // PERF_CUH
