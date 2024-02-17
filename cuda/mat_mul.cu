#include "mat_mul.cuh"

namespace Numeric::CUDA
{
__global__ void mat_mul(float *a, float *b, float *res,
                        unsigned num_rows_a, unsigned num_cols_a,
                        unsigned num_cols_b)
{
    auto const row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows_a && col < num_cols_b) {
        auto res_elem_val = 0.0f;
        for (auto idx = 0u; idx < num_cols_a; idx++) {
            res_elem_val += (a[row * num_cols_a + idx] * b[idx * num_cols_b + col]);
        }
        res[row * num_cols_b + col] = res_elem_val;
    }
}

auto constexpr TILE_WIDTH = 32u;

__global__ void mat_mul_shared_mem(float *a, float *b, float *res,
                                   unsigned num_rows_a, unsigned num_cols_a,
                                   unsigned num_cols_b)
{
    auto const row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    auto const col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    auto const num_rows_b = num_cols_a;
    auto const num_tiles = static_cast<unsigned>(
        ceil(static_cast<float>(max(num_rows_a, num_cols_b)) / TILE_WIDTH));
    for (auto tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {

        // Load the tiles into shared memory.
        auto const a_row = row;
        auto const b_col = col;
        auto const a_col = tile_idx * TILE_WIDTH + threadIdx.x;
        auto const b_row = tile_idx * TILE_WIDTH + threadIdx.y;

        if (a_row < num_rows_a && a_col < num_cols_a) {
            a_tile[threadIdx.y][threadIdx.x] = a[a_row * num_cols_a + a_col];
        }
        else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < num_rows_b && b_col < num_cols_b) {
            b_tile[threadIdx.y][threadIdx.x] = b[b_row * num_cols_b + b_col];
        }
        else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the result for the tiles.
        auto res_elem_val = 0.0f;
        for (auto idx = 0u; idx < TILE_WIDTH; idx++) {
            res_elem_val += (a_tile[threadIdx.y][idx] * b_tile[idx][threadIdx.x]);
        }

        // Store the result in the global memory.
        if (row < num_rows_a && col < num_cols_b) {
            res[row * num_cols_b + col] += res_elem_val;
        }
        __syncthreads();
    }
}

Eigen::MatrixXf matMul(Eigen::MatrixXf const &a,
                       Eigen::MatrixXf const &b,
                       bool use_shared_mem)
{
    if (a.cols() != b.rows()) {
        throw std::invalid_argument{"Invalid size for matrix multiplication."};
    }

    // Copy Eigen matrix objects to STL vector to copy to GPU.
    auto const a_vec = EigenUtils::toVec(a);
    auto const b_vec = EigenUtils::toVec(b);
    auto res_vec = std::vector<float>(a.rows() * b.cols(), -1.0f);

    // Allocate device GPU memory.
    auto a_vec_dev = static_cast<float *>(nullptr);
    auto b_vec_dev = static_cast<float *>(nullptr);
    auto res_vec_dev = static_cast<float *>(nullptr);
    cudaMalloc(reinterpret_cast<void **>(&a_vec_dev), a_vec.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&b_vec_dev), b_vec.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&res_vec_dev), res_vec.size() * sizeof(float));

    // Transfer input matrix elements to GPU.
    cudaMemcpy(a_vec_dev, a_vec.data(), a_vec.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_vec_dev, b_vec.data(), b_vec.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the kernel.
    auto const num_threads_per_block = dim3{TILE_WIDTH, TILE_WIDTH};
    auto const num_blocks_x = static_cast<unsigned>(
        std::ceil(static_cast<float>(b.cols()) /
            static_cast<float>(num_threads_per_block.x)));
    auto const num_blocks_y = static_cast<unsigned>(
        std::ceil(static_cast<float>(a.rows()) /
            static_cast<float>(num_threads_per_block.y)));
    auto const num_blocks = dim3{num_blocks_x, num_blocks_y};
    if (use_shared_mem) {
        mat_mul_shared_mem<<<num_blocks, num_threads_per_block>>>(a_vec_dev,
                                                                  b_vec_dev,
                                                                  res_vec_dev,
                                                                  static_cast<unsigned>(a.rows()),
                                                                  static_cast<unsigned>(a.cols()),
                                                                  static_cast<unsigned>(b.cols()));
    }
    else {
        mat_mul<<<num_blocks, num_threads_per_block>>>(a_vec_dev,
                                                       b_vec_dev,
                                                       res_vec_dev,
                                                       static_cast<unsigned>(a.rows()),
                                                       static_cast<unsigned>(a.cols()),
                                                       static_cast<unsigned>(b.cols()));
    }
    cudaMemcpy(res_vec.data(), res_vec_dev, res_vec.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return EigenUtils::toMat<float>(res_vec, a.rows(), b.cols());
}

} //Numeric namespace.