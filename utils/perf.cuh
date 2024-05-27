#ifndef PERF_CUH
#define PERF_CUH

#include <string>
#include <vector>
#include <map>

struct kernelPerfInfo
{
    float duration_ms;
};
using PerfTestResult = std::map<std::string, std::vector<float>>;


#endif //PERF_CUH


