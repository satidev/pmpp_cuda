#include "mat_mul.cuh"
#include "mat_mul_naive.cuh"
#include "mat_mul_cublas.cuh"
#include "mat_mul_tiled.cuh"
#include "sq_mat_mul_tiled_static_sm.cuh"
#include "sq_mat_mul_tiled_dynamic_sm.cuh"
#include "../../utils/dlib_utils.cuh"
#include <iostream>
#include <memory>
#include <gmock/gmock.h>

using namespace PMPP;

TEST(cudaMatMulTest, noThrowDueToValidSize)
{
    auto const first = dlib::matrix<float>(2, 3);
    auto const sec = dlib::matrix<float>(3, 4);
    auto const mat_mul = MatMul<float>{std::make_unique<MatMulNaive<float>>()};
    ASSERT_NO_THROW(mat_mul.run(first, sec));
}

TEST(cudaMatMulTest, throwsDueToInvalidSize)
{
    auto const first = dlib::matrix<float>(2, 3);
    auto const sec = dlib::matrix<float>(2, 3);
    auto const mat_mul = MatMul<float>{std::make_unique<MatMulNaive<float>>()};
    ASSERT_THROW(mat_mul.run(first, sec), std::invalid_argument);
}

TEST(cudaMatMulTest, validResMatSize)
{
    auto const first = dlib::matrix<float>(2, 3);
    auto const sec = dlib::matrix<float>(3, 4);
    auto const mat_mul = MatMul<float>{std::make_unique<MatMulNaive<float>>()};
    auto const res = std::get<0>(mat_mul.run(first, sec));
    ASSERT_EQ(res.nr(), first.nr());
    ASSERT_EQ(res.nc(), sec.nc());
}

TEST(cudaMatMulTest, squareMatMul)
{
    auto const N = 32;
    auto const first = DlibUtils::constMat(N, N, 1.0f);
    auto const sec = first;
    auto const res_exp = first * sec;

    auto mat_mul_vec = std::vector<MatMul<float>>{};
    mat_mul_vec.push_back(MatMul<float>{std::make_unique<MatMulNaive<float>>()});
    mat_mul_vec.push_back(MatMul<float>{std::make_unique<MatMulCuBlas<float>>()});
    mat_mul_vec.push_back(MatMul<float>{std::make_unique<SqMatMulTiledStaticSM<float>>()});
    mat_mul_vec.push_back(MatMul<float>{std::make_unique<SqMatMulTiledDynamicSM<float>>(16u)});

    for (auto const &mat_mul : mat_mul_vec)
    {
        auto const res = std::get<0>(mat_mul.run(first, sec));
        ASSERT_THAT(std::vector<float>(res.begin(), res.end()),
                    testing::ContainerEq(std::vector<float>(res_exp.begin(), res_exp.end())));
    }
}
