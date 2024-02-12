#include "mat_mul.cuh"

namespace Numeric::CUDA
{
__global__ void mat_mul_square(float *a, float *b, float *res,
                               unsigned num_cols)
{
    auto const row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col = blockIdx.x * blockDim.x + threadIdx.x;

    auto const num_rows = num_cols;
    if (row < num_rows && col < num_cols) {

        auto res_elem_val = 0.0f;
        for (auto idx = 0u; idx < num_cols; idx++) {
            res_elem_val += (a[row * num_cols + idx] * b[idx * num_cols + col]);
        }
        res[row * num_cols + col] = res_elem_val;
    }

}

Eigen::MatrixXf matMul(Eigen::MatrixXf const &a, Eigen::MatrixXf const &b)
{
    if ((a.rows() != a.cols()) || (b.rows() != b.cols())) {
        throw std::invalid_argument{
            "Currently matrix multiplication is supported only for square matrices."};
    }
    if ((a.rows() != b.rows() || (a.cols() != b.cols()))) {
        throw std::invalid_argument{"Invalid size for matrix multiplication."};
    }

    // Copy Eigen matrix objects to STL vector to copy to GPU.
    auto const a_vec = EigenUtils::toVec(a);
    auto const b_vec = EigenUtils::toVec(b);
    auto res_vec = std::vector<float>(a_vec.size());
    auto const vec_size_bytes = a_vec.size() * sizeof(float);

    // Allocate device GPU memory.
    auto a_vec_dev = static_cast<float *>(nullptr);
    auto b_vec_dev = static_cast<float *>(nullptr);
    auto res_vec_dev = static_cast<float *>(nullptr);
    cudaMalloc(reinterpret_cast<void **>(&a_vec_dev), vec_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&b_vec_dev), vec_size_bytes);
    cudaMalloc(reinterpret_cast<void **>(&res_vec_dev), vec_size_bytes);

    // Transfer input matrix elements to GPU.
    cudaMemcpy(a_vec_dev, a_vec.data(), vec_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_vec_dev, b_vec.data(), vec_size_bytes, cudaMemcpyHostToDevice);

    // Execute the kernel.
    auto const num_threads_per_block = dim3{16u, 16u};
    auto const num_blocks_x = static_cast<unsigned>(
        std::ceil(static_cast<float>(a.rows()) /
            static_cast<float>(num_threads_per_block.x)));
    auto const num_blocks = dim3{num_blocks_x, num_blocks_x};
    mat_mul_square<<<num_blocks, num_threads_per_block>>>(a_vec_dev,
                                                          b_vec_dev,
                                                          res_vec_dev,
                                                          static_cast<unsigned>(a.rows()));

    cudaMemcpy(res_vec.data(), res_vec_dev, vec_size_bytes, cudaMemcpyDeviceToHost);

    return EigenUtils::toMat<float>(res_vec, a.rows(), a.cols());
}

} //Numeric namespace.