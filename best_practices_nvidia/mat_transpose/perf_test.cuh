#ifndef PERF_TEST_CUH
#define PERF_TEST_CUH

#include "../../utils/perf.cuh"
#include <string>
#include <tuple>
#include <vector>


namespace BPNV
{
std::vector<PerfTestResult> transposePerfTest(unsigned num_rep = 10u);
}// BPNV namespace.
#endif //PERF_TEST_CUH
