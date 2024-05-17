#ifndef MAT_MUL_TILED_CUH
#define MAT_MUL_TILED_CUH

#include "mat_mul_impl.cuh"
#include "../../utils/check_error.cuh"
#include "../../utils/dev_config.cuh"
#include <stdexcept>

namespace PMPP
{
template<typename T>
class MatMulTiled: public MatMulImpl<T>
{
public:
    void launchKernel(T const *first, T const *sec, T *res,
                      unsigned num_rows_first,
                      unsigned num_cols_first,
                      unsigned num_cols_sec) final;
};

__global__ void matMulTiledKernel(float const *a, float const *b, float *res,
                                    unsigned num_rows_a, unsigned num_cols_a,
                                    unsigned num_cols_b)
{
    // It is assumed that blockDim.x == blockDim.y.
    auto const tile_width = blockDim.x;
    auto const row = blockIdx.y * tile_width + threadIdx.y;
    auto const col = blockIdx.x * tile_width + threadIdx.x;

    extern __shared__ float shared_mem[];
    auto a_tile = reinterpret_cast<float *>(shared_mem);
    auto b_tile = reinterpret_cast<float *>(shared_mem + tile_width * tile_width);

    auto const num_rows_b = num_cols_a;
    auto const num_tiles = static_cast<unsigned>(
        ceil(static_cast<float>(max(num_rows_a, num_cols_b)) /
            static_cast<float>(tile_width)));
    for (auto tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {

        // Load the tiles into shared memory.
        auto const a_row = row;
        auto const b_col = col;
        auto const a_col = tile_idx * tile_width + threadIdx.x;
        auto const b_row = tile_idx * tile_width + threadIdx.y;

        if (a_row < num_rows_a && a_col < num_cols_a) {
            a_tile[threadIdx.y * tile_width + threadIdx.x] = a[a_row * num_cols_a + a_col];
        }
        else {
            a_tile[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }
        if (b_row < num_rows_b && b_col < num_cols_b) {
            b_tile[threadIdx.y * tile_width + threadIdx.x] = b[b_row * num_cols_b + b_col];
        }
        else {
            b_tile[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the result for the tiles.
        auto res_elem_val = 0.0f;
        for (auto idx = 0u; idx < tile_width; idx++) {
            res_elem_val += (a_tile[threadIdx.y * tile_width + idx]
                * b_tile[idx * tile_width + threadIdx.x]);
        }

        // Store the result in the global memory.
        if (row < num_rows_a && col < num_cols_b) {
            res[row * num_cols_b + col] += res_elem_val;
        }
        __syncthreads();
    }
}

template<typename T>
void MatMulTiled<T>::launchKernel(T const *first,
                                  T const *sec,
                                  T *res,
                                  unsigned num_rows_first,
                                  unsigned num_cols_first,
                                  unsigned num_cols_sec)
{
    auto const block_size = dim3{16u, 16u};
    auto const num_block_x = (num_cols_sec + block_size.x - 1u) / block_size.x;
    auto const num_block_y = (num_rows_first + block_size.y - 1u) / block_size.y;
    auto const grid_size = dim3{num_block_x, num_block_y};
    auto const &dev_config = DeviceConfigSingleton::getInstance();
    auto const shared_mem_per_blk = dev_config.getDevProps(0).max_shared_mem_per_block;

    if (block_size.x != block_size.y) {
        throw std::invalid_argument{
            "The number of threads per block in each dimension must be equal."};
    }
    auto const tile_width = block_size.x;
    auto const shared_mem_size = 2 * tile_width * tile_width * sizeof(T);
    if (shared_mem_size > shared_mem_per_blk) {
        throw std::runtime_error{"Shared memory size exceeds the limit."};
    }
    matMulTiledKernel<<<grid_size, block_size>>>(first, sec, res,
                                                 num_rows_first,
                                                 num_cols_first,
                                                 num_cols_sec);
}

}// PMPP namespace.
#endif //MAT_MUL_TILED_CUH


