#include "mat_mul.cuh"
#include "dev_config.cuh"

namespace Numeric::CUDA
{
__global__ void mat_mul(float const *a, float const *b, float *res,
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

__global__ void mat_mul_shared_mem(float const *a, float const *b, float *res,
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

    // Check the device has enough global memory to store all vectors.
    auto const &dev_config = DeviceConfigSingleton::getInstance();
    if ((a_vec.size() + b_vec.size() + res_vec.size()) * sizeof(float) >
        dev_config.getDevProps(0).global_mem_size) {
        throw std::runtime_error{"Insufficient global memory on the device."};
    }


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
    auto const shared_mem_per_blk = dev_config.getDevProps(0).max_shared_mem_per_block;

    auto const block_size = dim3{16u, 16u};
    auto const num_block_x = (static_cast<unsigned>(b.cols()) + block_size.x - 1u) / block_size.x;
    auto const num_block_y = (static_cast<unsigned>(a.rows()) + block_size.y - 1u) / block_size.y;
    auto const grid_size = dim3{num_block_x, num_block_y};

    if (use_shared_mem) {
        if(block_size.x != block_size.y) {
            throw std::invalid_argument{"The number of threads per block in each dimension must be equal."};
        }
        auto const tile_width = block_size.x;
        auto const shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
        if (shared_mem_size > shared_mem_per_blk) {
            throw std::runtime_error{"Shared memory size exceeds the limit."};
        }
        mat_mul_shared_mem<<<grid_size, block_size, shared_mem_size>>>(
            a_vec_dev, b_vec_dev, res_vec_dev,
            static_cast<unsigned>(a.rows()),
            static_cast<unsigned>(a.cols()),
            static_cast<unsigned>(b.cols()));
    }
    else {
        mat_mul<<<grid_size, block_size>>>(a_vec_dev,
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