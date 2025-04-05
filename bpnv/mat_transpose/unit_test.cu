#include "transpose.cuh"
#include "trans_impl_naive.cuh"
#include "trans_impl_sm.cuh"
#include "trans_impl_sm_padding.cuh"
#include "trans_impl_sm_swizzling.cuh"
#include "../../utils/dlib_utils.cuh"
#include <gmock/gmock.h>

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
    auto constexpr N = 32;
    auto vec = std::vector<float>(N * N);
    std::iota(vec.begin(), vec.end(), 0.0f);
    auto const input = dlib::mat(vec.data(), N, N);
    auto const output_exp = dlib::trans(input);

    auto trans_vec = std::vector<Transpose<float>>{};
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplNaive<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSM<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSMPadding<float>>()});
    trans_vec.push_back(Transpose<float>{std::make_unique<TransImplSMSwizzling<float>>()});

    for (auto const &trans : trans_vec)
    {
        auto const output = std::get<0>(trans.run(input));
        ASSERT_THAT(std::vector<float>(output.begin(), output.end()),
                    testing::ContainerEq(std::vector<float>(output_exp.begin(), output_exp.end())));
    }
}
