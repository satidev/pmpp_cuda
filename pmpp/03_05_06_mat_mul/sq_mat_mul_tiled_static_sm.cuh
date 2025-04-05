#ifndef SQ_MAT_MUL_TILED_STATIC_SM_CUH
#define SQ_MAT_MUL_TILED_STATIC_SM_CUH

#include "mat_mul_impl_strategy.cuh"
#include "../../utils/dev_config.cuh"
#include <stdexcept>

namespace PMPP
{
    // Static shared memory version of the tiled matrix multiplication
    // for square matrices.
    template <typename T>
    class SqMatMulTiledStaticSM : public MatMulImplStrategy<T>
    {
    public:
        void launchKernel(T const *first, T const *sec, T *res,
                          unsigned num_rows_first,
                          unsigned num_cols_first,
                          unsigned num_cols_sec) final;
    };
#define TILE_SIZE 16u

    template <typename T>
    __global__ void SqMatMulTiledStaticSMKernel(T const *first, T const *sec, T *res,
                                                unsigned N)
    {
        auto const row = blockIdx.y * blockDim.y + threadIdx.y;
        auto const col = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ T first_tile[TILE_SIZE][TILE_SIZE];
        __shared__ T sec_tile[TILE_SIZE][TILE_SIZE];

        auto const num_tiles = (N + TILE_SIZE - 1u) / TILE_SIZE;

        if (row < N && col < N)
        {
            auto res_elem_val = T{};
            for (auto tile_idx = 0u; tile_idx < num_tiles; tile_idx++)
            {

                auto const first_col = tile_idx * TILE_SIZE + threadIdx.x;
                auto const sec_row = tile_idx * TILE_SIZE + threadIdx.y;

                first_tile[threadIdx.y][threadIdx.x] = first[row * N + first_col];
                sec_tile[threadIdx.y][threadIdx.x] = sec[sec_row * N + col];
                __syncthreads();

                for (auto idx = 0u; idx < TILE_SIZE; idx++)
                {
                    res_elem_val += first_tile[threadIdx.y][idx] * sec_tile[idx][threadIdx.x];
                }
                __syncthreads();
            }
            res[row * N + col] = res_elem_val;
        }
    }

    template <typename T>
    void SqMatMulTiledStaticSM<T>::launchKernel(T const *first,
                                                T const *sec,
                                                T *res,
                                                unsigned num_rows_first,
                                                unsigned num_cols_first,
                                                unsigned num_cols_sec)
    {
        // Make sure that the matrices are square.
        if (num_rows_first != num_cols_first || num_cols_first != num_cols_sec)
        {
            throw std::invalid_argument{"The matrices are not square."};
        }

        // Make sure that the number of rows/ columns are multiples 32.
        if (num_rows_first % TILE_SIZE != 0)
        {
            throw std::invalid_argument{"The number of rows is not a multiple of 32."};
        }

        auto const num_blocks = (num_cols_sec + TILE_SIZE - 1u) / TILE_SIZE;

        auto const block_size = dim3{TILE_SIZE, TILE_SIZE};
        auto const grid_size = dim3{num_blocks, num_blocks};
        auto const &dev_config = DeviceConfigSingleton::getInstance();
        auto const shared_mem_per_blk = dev_config.getDevProps(0).max_shared_mem_per_block;

        if (block_size.x != block_size.y)
        {
            throw std::invalid_argument{
                "The number of threads per block in each dimension must be equal."};
        }
        auto const shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(T);
        if (shared_mem_size > shared_mem_per_blk)
        {
            throw std::runtime_error{"Shared memory size exceeds the limit."};
        }
        SqMatMulTiledStaticSMKernel<<<grid_size, block_size>>>(first, sec, res,
                                                               num_rows_first);
        checkErrorKernel("SqMatMulTiledStaticSMKernel", true);
    }

} // PMPP namespace.

#endif // SQ_MAT_MUL_TILED_STATIC_SM_CUH
