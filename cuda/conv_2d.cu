#include "conv_2d.cuh"
#include "../cpp/eigen_utils.h"

namespace Numeric::CUDA
{
__global__ void conv_kern_2d(float const *data,
                             float const *filter,
                             float *res,
                             unsigned num_rows,
                             unsigned num_cols,
                             unsigned filter_radius);


__global__ void conv_kern_2d(float const *data,
                             float const *filter,
                             float *res,
                             unsigned num_rows,
                             unsigned num_cols,
                             unsigned filter_radius)
{
    auto const row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {

        auto const filter_size = 2u * filter_radius + 1u;

        // Element by element multiplication and accumulation.
        auto sum = 0.0f;

        for (auto filt_row = 0u; filt_row < filter_size; ++filt_row) {
            for (auto filt_col = 0u; filt_col < filter_size; ++filt_col) {
                // Flipped indices.
                auto const data_row = static_cast<int>(row) - static_cast<int>(filt_row) +
                                    static_cast<int>(filter_radius);
                auto const data_col = static_cast<int>(col) - static_cast<int>(filt_col) +
                                    static_cast<int>(filter_radius);
                if (data_row >= 0 && data_row < num_rows && data_col >= 0 && data_col < num_cols) {
                    sum += data[data_row * num_cols + data_col] * filter[filt_row * filter_size + filt_col];
                }
            }
        }
        res[row * num_cols + col] = sum;
    }

}


Eigen::MatrixXf conv2D(Eigen::MatrixXf const &data,
                       Eigen::MatrixXf const &filter,
                       bool use_shared_mem)
{
    // Check the data size.
    if (data.size() == 0u) {
        throw std::invalid_argument("Data size is zero.");
    }

    // Check the filter size.
    if (filter.size() == 0u) {
        throw std::invalid_argument("Filter size is zero.");
    }

    // Make sure that the 2D filter is square.
    if (filter.rows() != filter.cols()) {
        throw std::invalid_argument("Filter is not square.");
    }

    // Check the filter size is odd.
    if (filter.rows() % 2u == 0u) {
        throw std::invalid_argument("Filter size is not odd.");
    }

    // Copy Eigen matrix objects to STL vector to copy to GPU.
    auto const data_vec = EigenUtils::toVec(data);
    auto const filter_vec = EigenUtils::toVec(filter);
    auto res_vec = std::vector<float>(data.size(), 0.0f);

    // Copy the data and filter to the GPU.
    auto const data_size_bytes = data.size() * sizeof(float);
    auto const filter_size_bytes = filter.size() * sizeof(float);
    auto const res_size_bytes = res_vec.size() * sizeof(float);

    auto d_data = static_cast<float*>(nullptr);
    auto d_filter = static_cast<float*>(nullptr);
    auto d_res = static_cast<float*>(nullptr);
    cudaMalloc(reinterpret_cast<void**>(&d_data), data_size_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_filter), filter_size_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_res), res_size_bytes);

    // Transfer data from the host to the device.
    cudaMemcpy(d_data, data_vec.data(), data_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter_vec.data(), filter_size_bytes, cudaMemcpyHostToDevice);

    // Launch the kernel.
    if(use_shared_mem) {

    }
    else {
        auto const block_size = dim3{16u, 16u};
        auto const num_block_x = (static_cast<unsigned>(data.cols()) + block_size.x - 1u) / block_size.x;
        auto const num_block_y = (static_cast<unsigned>(data.rows()) + block_size.y - 1u) / block_size.y;
        auto const grid_size = dim3{num_block_x, num_block_y};

        auto const filter_radius = static_cast<unsigned>(filter.rows() / 2u);
        conv_kern_2d<<<grid_size, block_size>>>(d_data, d_filter, d_res,
                                                static_cast<unsigned>(data.rows()),
                                                static_cast<unsigned>(data.cols()),
                                                filter_radius);
    }
    // Transfer the result back to the host.
    cudaMemcpy(res_vec.data(), d_res, res_size_bytes, cudaMemcpyDeviceToHost);

    return EigenUtils::toMat<float>(res_vec, data.rows(), data.cols());
}
}// Numeric::CUDA namespace.
