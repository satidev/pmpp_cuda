#include "transpose.cuh"
#include "../../utils/dlib_utils.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm_padding.cuh"
#include "trans_impl_sm_swizzling.cuh"
#include "trans_impl_sm.cuh"
#include <string>

namespace BPNV
{
PerfTestResult transposePerfTest(unsigned num_rep)
{
    std::cout << "Performance test for matrix transpose: start" << std::endl;

    auto const N = 32 * 600;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    auto const input = DlibUtils::constMat(N, N, 1.0f);

    auto trans_vec = std::vector<std::pair<Transpose<float>, std::string>>{};
    trans_vec.push_back(std::make_pair(Transpose<float>{std::make_unique<TransImplNaive<float>>()},
                                       "naive"));
    trans_vec.push_back(std::make_pair(Transpose<float>{std::make_unique<TransImplSM<float>>()},
                                       "sm"));
    trans_vec.push_back(std::make_pair(
        Transpose<float>{std::make_unique<TransImplSMPadding<float>>()},
        "sm-padding"));

    trans_vec.push_back(std::make_pair(
        Transpose<float>{std::make_unique<TransImplSMSwizzling<float>>()},
        "sm-swizzling"));

    auto perf_res = PerfTestResult{};
    for (auto const &[trans, desc]: trans_vec) {

        auto perf_vec = std::vector<float>{};
        for (auto run_idx = 0u; run_idx < num_rep; ++run_idx) {
            auto const res = trans.run(input);
            perf_vec.emplace_back(std::get<1>(res).exec_duration.count());
        }
        perf_res[desc] = perf_vec;
    }
    std::cout << "Performance test for matrix transpose: end" << std::endl;

    return perf_res;
}
}// BPNV namespace.

