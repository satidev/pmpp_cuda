#include "mat_mul.cuh"
#include "../utils/dev_config.cuh"
#include "../utils/dev_vector.cuh"
#include <cublas_v2.h>
#include "../utils/dev_timer.cuh"
#include "mat_mul_naive.cuh"
#include "mat_mul_cublas.cuh"
#include "sq_mat_mul_tiled_static_sm.cuh"
#include "sq_mat_mul_tiled_dynamic_sm.cuh"
#include "../utils/dlib_utils.cuh"
#include <dlib/matrix.h>
#include <string>

namespace PMPP
{
void matMulPerfTest()
{
    std::cout << "Performance test for matrix multiplication: start" << std::endl;
    auto const N = 32 * 300;
    std::cout << "Matrix size: "<< N <<"x" << N << std::endl;
    auto const first = DlibUtils::constMat(N, N, 1.0f);
    auto const sec = first;

    auto mat_mul_vec = std::vector<std::pair<MatMul<float>, std::string>>{};
    mat_mul_vec.push_back(std::make_pair(MatMul<float>{std::make_unique<MatMulNaive<float>>()},
                                         "Naive"));
    mat_mul_vec.push_back(std::make_pair(MatMul<float>{std::make_unique<MatMulCuBlas<float>>()},
                                         "CuBlas"));
    mat_mul_vec.push_back(
        std::make_pair(MatMul<float>{std::make_unique<SqMatMulTiledStaticSM<float>>()},
                       "Static shared memory"));
    mat_mul_vec.push_back(
        std::make_pair(MatMul<float>{std::make_unique<SqMatMulTiledDynamicSM<float>>(32u)},
                       "Dynamic shared memory"));

    for (auto const &[mat_mul, desc]: mat_mul_vec) {
        auto const res = mat_mul.run(first, sec);
        std::cout << desc << ": " << std::get<1>(res).exec_duration.count() << " milli seconds."
                  << std::endl;
    }
    std::cout << "Performance test for matrix multiplication: end" << std::endl;
}

} //PMPP namespace.