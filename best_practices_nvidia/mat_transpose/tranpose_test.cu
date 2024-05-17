#include <gmock/gmock.h>
#include "transpose.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm.cuh"
#include "trans_impl_sm_no_bank_conflict.cuh"
#include "../../pmpp/utils/dlib_utils.cuh"

using namespace BPNV;

TEST(cudaTransposeTest, validResMatSize)
{
    auto const input = dlib::matrix<float>(2, 3);
    auto const trans = Transpose<float>{std::make_unique<TransImplNaive<float>>()};
    auto const res = std::get<0>(trans.run(input));
    ASSERT_EQ(res.nr(), input.nc());
    ASSERT_EQ(res.nc(), input.nr());
}

TEST(cudaTransposeTest, squareMatTranspose)
{
    auto const N = 32;
    auto const input = DlibUtils::constMat(N, N, 1.0f);
    auto const output_exp = dlib::trans(input);

    auto trans_vec = std::vector<Transpose<float>>{};
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplNaive<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSM<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSMNoBankConflict<float>>()});

    for (auto const &trans: trans_vec) {
        auto const output = std::get<0>(trans.run(input));
        ASSERT_THAT(std::vector<float>(output.begin(), output.end()),
                    testing::ContainerEq(std::vector<float>(output_exp.begin(), output_exp.end())));
    }
}
