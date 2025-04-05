#ifndef MAT_MUL_NAIVE_CUH
#define MAT_MUL_NAIVE_CUH

#include "mat_mul_impl_strategy.cuh"
#include "../../utils/check_error.cuh"

namespace PMPP
{
    template <typename T>
    class MatMulNaive : public MatMulImplStrategy<T>
    {
    public:
        void launchKernel(T const *first, T const *sec, T *res,
                          unsigned num_rows_first,
                          unsigned num_cols_first,
                          unsigned num_cols_sec) final;
    };

    template <typename T>
    __global__ void matMulNaiveKernel(T const *first, T const *sec, T *res,
                                      unsigned num_rows_first,
                                      unsigned num_cols_first,
                                      unsigned num_cols_sec)
    {
        auto const row = blockIdx.y * blockDim.y + threadIdx.y;
        auto const col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < num_rows_first && col < num_cols_sec)
        {
            auto res_elem_val = 0.0f;
            for (auto idx = 0u; idx < num_cols_first; idx++)
            {
                res_elem_val += (first[row * num_cols_first + idx] * sec[idx * num_cols_sec + col]);
            }
            res[row * num_cols_sec + col] = res_elem_val;
        }
    }

    template <typename T>
    void MatMulNaive<T>::launchKernel(T const *first, T const *sec, T *res,
                                      unsigned num_rows_first,
                                      unsigned num_cols_first,
                                      unsigned num_cols_sec)
    {
        auto const block_size = dim3{16u, 16u};
        auto const num_block_x = (num_cols_sec + block_size.x - 1u) / block_size.x;
        auto const num_block_y = (num_rows_first + block_size.y - 1u) / block_size.y;
        auto const grid_size = dim3{num_block_x, num_block_y};

        matMulNaiveKernel<<<grid_size, block_size>>>(first, sec, res,
                                                     num_rows_first,
                                                     num_cols_first,
                                                     num_cols_sec);
        checkErrorKernel("matMulKernel", true);
    }

} // PMPP namespace.

#endif // MAT_MUL_NAIVE_CUH
