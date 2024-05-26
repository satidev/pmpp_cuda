#ifndef PERF_CUH
#define PERF_CUH

#include <string>
#include <vector>

struct PerfInfo
{
    float kernel_duration_ms;
};

struct PerfTestResult
{
    std::string label;
    // Performance metric of each run or repetition.
    std::vector<float> metric_vals;
};

#endif //PERF_CUH


