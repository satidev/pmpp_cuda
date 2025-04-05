#ifndef SQ_MAT_MUL_TILED_DYNAMIC_SM_CUH
#define SQ_MAT_MUL_TILED_DYNAMIC_SM_CUH

#include "mat_mul_impl_strategy.cuh"
#include "../../utils/dev_config.cuh"
#include "../../utils/check_error.cuh"
#include <stdexcept>

namespace PMPP
{
    // Static shared memory version of the tiled matrix multiplication
    // for square matrices.
    template <typename T>
    class SqMatMulTiledDynamicSM : public MatMulImplStrategy<T>
    {
    private:
        unsigned tile_size_;

    public:
        explicit SqMatMulTiledDynamicSM(unsigned tile_size)
            : tile_size_{tile_size}
        {
        }
        void launchKernel(T const *first, T const *sec, T *res,
                          unsigned num_rows_first,
                          unsigned num_cols_first,
                          unsigned num_cols_sec) final;
    };

    template <typename T>
    __global__ void SqMatMulTiledDynamicSMKernel(T const *first, T const *sec, T *res,
                                                 unsigned N)
    {
        auto const row = blockIdx.y * blockDim.y + threadIdx.y;
        auto const col = blockIdx.x * blockDim.x + threadIdx.x;
        auto const tile_size = blockDim.x;

        extern __shared__ T shared_mem[];
        auto first_tile = reinterpret_cast<T *>(shared_mem);
        auto sec_tile = reinterpret_cast<T *>(shared_mem + tile_size * tile_size);

        auto const num_tiles = (N + tile_size - 1u) / tile_size;

        if (row < N && col < N)
        {
            auto res_elem_val = T{};
            for (auto tile_idx = 0u; tile_idx < num_tiles; tile_idx++)
            {

                auto const first_col = tile_idx * tile_size + threadIdx.x;
                auto const sec_row = tile_idx * tile_size + threadIdx.y;

                first_tile[threadIdx.y * tile_size + threadIdx.x] = first[row * N + first_col];
                sec_tile[threadIdx.y * tile_size + threadIdx.x] = sec[sec_row * N + col];
                __syncthreads();

                for (auto idx = 0u; idx < tile_size; idx++)
                {
                    res_elem_val += first_tile[threadIdx.y * tile_size + idx] *
                                    sec_tile[idx * tile_size + threadIdx.x];
                }
                __syncthreads();
            }
            res[row * N + col] = res_elem_val;
        }
    }

    template <typename T>
    void SqMatMulTiledDynamicSM<T>::launchKernel(T const *first,
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
        if (num_rows_first % tile_size_ != 0)
        {
            throw std::invalid_argument{"The number of rows is not a multiple of 32."};
        }

        auto const num_blocks = (num_cols_sec + tile_size_ - 1u) / tile_size_;

        auto const block_size = dim3{tile_size_, tile_size_};
        auto const grid_size = dim3{num_blocks, num_blocks};
        auto const &dev_config = DeviceConfigSingleton::getInstance();
        auto const shared_mem_per_blk = dev_config.getDevProps(0).max_shared_mem_per_block;

        if (block_size.x != block_size.y)
        {
            throw std::invalid_argument{
                "The number of threads per block in each dimension must be equal."};
        }
        auto const shared_mem_size = 2 * tile_size_ * tile_size_ * sizeof(T);
        if (shared_mem_size > shared_mem_per_blk)
        {
            throw std::runtime_error{"Shared memory size exceeds the limit."};
        }
        SqMatMulTiledDynamicSMKernel<<<grid_size, block_size, shared_mem_size>>>(first, sec, res,
                                                                                 num_rows_first);
        checkErrorKernel("SqMatMulTiledStaticSMKernel", true);
    }

} // PMPP namespace.

#endif // SQ_MAT_MUL_TILED_DYNAMIC_SM_CUH
