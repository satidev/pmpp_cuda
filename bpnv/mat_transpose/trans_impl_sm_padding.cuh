#ifndef TRANS_IMPL_SM_PADDING_CUH
#define TRANS_IMPL_SM_PADDING_CUH

#include "trans_impl_strategy.cuh"
#include "../../utils/check_error.cuh"

namespace BPNV
{
// Transpose matrix using shared memory with extra padded column
// to avoid bank conflicts.
template<typename T>
class TransImplSMPadding: public TransImplStrategy<T>
{
public:
    virtual void launchKernel(T const *input, T *output,
                              unsigned num_rows_input, unsigned num_cols_input) const final;

};
#define TILE_SIZE 16u
template<typename T>
__global__ void transSMPaddedKernel(T const *input, T *output,
                                    unsigned num_rows_input, unsigned num_cols_input)
{
    // Bank conflict is arisen because the number of threads in a warp is 32
    // and the shared memory is divided into 32 banks.
    // During the reading of each column in shared memory,
    // the threads in warp access the same bank serializing the memory requests.
    // To avoid bank conflict and facilitate concurrent memory access,
    // we increase the number of columns in shared memory by 1.
    // By following the modulo operation, while accessing the element of each column
    // the threads of a warp access different banks
    // avoiding bank conflicts.
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

    auto const row_input = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col_input = blockIdx.x * blockDim.x + threadIdx.x;


    if (row_input < num_rows_input && col_input < num_cols_input) {
        // Coalesced reading of input matrix elements row by row.
        tile[threadIdx.y][threadIdx.x] = input[row_input * num_cols_input + col_input];
    }

    __syncthreads();

    auto const row_output = blockIdx.x * blockDim.x + threadIdx.y;
    auto const col_output = blockIdx.y * blockDim.y + threadIdx.x;
    auto const num_rows_output = num_cols_input;
    auto const num_cols_output = num_rows_input;
    if (row_output < num_rows_output && col_output < num_cols_output) {
        // Uncoalesced reading from the shared memory (column by column).
        // But coalesced writing to the output matrix (row by row).
        output[row_output * num_cols_output + col_output] = tile[threadIdx.x][threadIdx.y];
    }
}

template<typename T>
void TransImplSMPadding<T>::launchKernel(const T *input,
                                         T *output,
                                         unsigned int num_rows_input,
                                         unsigned int num_cols_input) const
{
    auto const block_size = dim3{TILE_SIZE, TILE_SIZE};
    auto const num_block_x = (num_cols_input + block_size.x - 1u) / block_size.x;
    auto const num_block_y = (num_rows_input + block_size.y - 1u) / block_size.y;
    auto const grid_size = dim3{num_block_x, num_block_y};

    transSMPaddedKernel<<<grid_size, block_size>>>(input, output, num_rows_input, num_cols_input);
    checkErrorKernel("transSMPaddedKernel", true);
}

}// BPNV namespace.

#endif //TRANS_IMPL_SM_PADDING_CUH


