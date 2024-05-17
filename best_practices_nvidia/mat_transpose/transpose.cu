#include "transpose.cuh"
#include "../../pmpp/utils/dlib_utils.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm_no_bank_conflict.cuh"
#include "trans_impl_sm.cuh"

namespace BPNV
{
void tranposeExample()
{
    auto const N = 32 * 600;
    auto const input = DlibUtils::constMat(N, N, 1.0f);

    auto trans_vec = std::vector<Transpose<float>>{};
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplNaive<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSM<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSMNoBankConflict<float>>()});

    for (auto const &trans: trans_vec) {
        auto const res = trans.run(input);
        std::cout << "Time taken: " << std::get<1>(res).kernel_duration_ms << " sec" << std::endl;
    }
}
}// BPNV namespace.
