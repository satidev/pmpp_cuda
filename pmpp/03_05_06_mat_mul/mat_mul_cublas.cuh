#ifndef MAT_MUL_CUBLAS_CUH
#define MAT_MUL_CUBLAS_CUH

#include "mat_mul_impl_strategy.cuh"
#include <cublas_v2.h>

namespace PMPP
{
    template <typename T>
    class MatMulCuBlas : public MatMulImplStrategy<T>
    {
    public:
        void launchKernel(T const *first, T const *sec, T *res,
                          unsigned num_rows_first,
                          unsigned num_cols_first,
                          unsigned num_cols_sec) final;
    };
    template <typename T>
    void MatMulCuBlas<T>::launchKernel(T const *first,
                                       T const *sec,
                                       T *res,
                                       unsigned num_rows_first,
                                       unsigned num_cols_first,
                                       unsigned num_cols_sec)
    {
        cublasHandle_t handle;
        cublasCreate_v2(&handle);

        auto const alpha = static_cast<T>(1.0f);
        auto const beta = static_cast<T>(0.0f);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_cols_sec, num_rows_first, num_cols_first,
                    &alpha, sec, num_cols_sec, first, num_cols_first, &beta, res, num_cols_sec);

        cublasDestroy_v2(handle);
    }

} // PMPP namespace.

#endif // MAT_MUL_CUBLAS_CUH
