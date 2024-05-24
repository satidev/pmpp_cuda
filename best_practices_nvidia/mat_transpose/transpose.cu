#include "transpose.cuh"
#include "../../utils/dlib_utils.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm_padding.cuh"
#include "trans_impl_sm_swizzling.cuh"
#include "trans_impl_sm.cuh"
#include <string>

namespace BPNV
{
void transposePerfTest()
{
    std::cout << "Performance test for matrix transpose: start" << std::endl;

    auto const N = 32 * 600;
    std::cout << "Matrix size: "<< N <<"x" << N << std::endl;
    auto const input = DlibUtils::constMat(N, N, 1.0f);

    auto trans_vec = std::vector<std::pair<Transpose<float>, std::string>>{};
    trans_vec.push_back(std::make_pair(Transpose<float>{std::make_unique<TransImplNaive<float>>()},
                                       "Naive"));
    trans_vec.push_back(std::make_pair(Transpose<float>{std::make_unique<TransImplSM<float>>()},
                                       "Shared memory"));
    trans_vec.push_back(std::make_pair(
        Transpose<float>{std::make_unique<TransImplSMPadding<float>>()},
        "Shared memory without bank conflicts (padding)"));

    trans_vec.push_back(std::make_pair(
        Transpose<float>{std::make_unique<TransImplSMSwizzling<float>>()},
        "Shared memory without bank conflict (swizzling)"));

    auto time_vec = std::vector<float>{};
    for (auto const &[trans, desc]: trans_vec) {
        auto const res = trans.run(input);
        auto const time_taken_sec = std::get<1>(res).kernel_duration_ms;
        time_vec.push_back(time_taken_sec);
        std::cout << desc << ": " << time_taken_sec << " milli seconds." << std::endl;
    }

    std::cout << "Performance boost compared to naive implementation:" << std::endl;
    auto const ref_time = time_vec.front();
    for (auto i = 1u; i < time_vec.size(); ++i) {
        auto const perf_boost = (ref_time - time_vec[i]) *100/ ref_time;
        std::cout << trans_vec[i].second << ": " << perf_boost << "%" << std::endl;
    }

    std::cout << "Performance test for matrix transpose: end" << std::endl;
}
}// BPNV namespace.
