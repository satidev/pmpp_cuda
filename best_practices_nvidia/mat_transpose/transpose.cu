#include "transpose.cuh"
#include "../../pmpp/utils/dlib_utils.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm_no_bank_conflict.cuh"
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
        Transpose<float>{std::make_unique<TransImplSMNoBankConflict<float>>()},
        "Shared memory without bank conflict"));

    for (auto const &[trans, desc]: trans_vec) {
        auto const res = trans.run(input);
        std::cout << desc << ": " << std::get<1>(res).kernel_duration_ms << " milli seconds." << std::endl;
    }

    std::cout << "Performance test for matrix transpose: end" << std::endl;
}
}// BPNV namespace.
