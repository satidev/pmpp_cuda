#ifndef TRANS_IMPL_NAIVE_CUH
#define TRANS_IMPL_NAIVE_CUH

#include "trans_impl_strategy.cuh"
#include "../../pmpp/utils/check_error.cuh"

namespace BPNV
{
template<typename T>
class TransImplNaive: public TransImplStrategy<T>
{
public:
    virtual void launchKernel(T const *input, T *output,
                              unsigned num_rows_input, unsigned num_cols_input) const final;
};

template<typename T>
__global__ void transNaiveKernel(T const *input, T *output,
                                 unsigned num_rows_input, unsigned num_cols_input)
{
    auto const row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows_input && col < num_cols_input) {
        // Coalesced reading of input matrix elements.
        // However, the writing of elements to output matrix
        // is not coalesced but strided leads to performance penalty.
        output[col * num_rows_input + row] = input[row * num_cols_input + col];
    }
}


template<typename T>
void TransImplNaive<T>::launchKernel(T const *input,
                                     T *output,
                                     unsigned num_rows_input,
                                     unsigned num_cols_input) const
{
    auto const block_size = dim3{16u, 16u};
    auto const num_block_x = (num_cols_input + block_size.x - 1u) / block_size.x;
    auto const num_block_y = (num_rows_input + block_size.y - 1u) / block_size.y;
    auto const grid_size = dim3{num_block_x, num_block_y};

    transNaiveKernel<<<grid_size, block_size>>>(input, output, num_rows_input, num_cols_input);
    checkErrorKernel("transNaiveKernel", true);
}

}// BPNV namespace.

#endif //TRANS_IMPL_NAIVE_CUH


