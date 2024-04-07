#include "stencil_2d.cuh"
#include "../cpp/eigen_utils.h"
#include <stdexcept>
#include "check_error.cuh"

namespace Numeric::CUDA
{

__host__ __device__ unsigned elemIndex(unsigned row, unsigned col, unsigned num_cols)
{
    return row * num_cols + col;
}
__global__ void sum_stencil_kern(float const *ip, float *op,
                                 unsigned num_rows,
                                 unsigned num_cols)
{
    auto const row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    auto const col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if ((row >= 0) && (row < num_rows) &&
        (col >= 0) && (col < num_cols)) {

        if ((row > 0) && (row < (num_rows - 1u)) &&
            (col > 0) && (col < (num_cols - 1u))) {

            op[elemIndex(row, col, num_cols)] =
                ip[elemIndex(row, col, num_cols)] +
                    ip[elemIndex(row - 1u, col, num_cols)] +
                    ip[elemIndex(row + 1u, col, num_cols)] +
                    ip[elemIndex(row, col - 1u, num_cols)] +
                    ip[elemIndex(row, col + 1u, num_cols)];

        }
        else {// Copy the halo values.
            op[elemIndex(row, col, num_cols)] = ip[elemIndex(row, col, num_cols)];
        }
    }
}

__global__ void sum_stencil_kern_sm(float const *ip, float *op,
                                    unsigned num_rows,
                                    unsigned num_cols)
{
    auto const ip_tile_size = static_cast<int>(blockDim.x);
    auto const op_tile_size = ip_tile_size - 2;

    auto const row = static_cast<int>(blockIdx.y * op_tile_size + threadIdx.y) - 1;
    auto const col = static_cast<int>(blockIdx.x * op_tile_size + threadIdx.x) - 1;

    // Copy to shared memory.
    extern __shared__ float sm[];
    if (row >= 0 && row < num_rows && col >= 0 && col < num_cols) {
        sm[elemIndex(threadIdx.y, threadIdx.x, ip_tile_size)] =
            ip[elemIndex(row, col, num_cols)];
    }
    else {
        sm[elemIndex(threadIdx.y, threadIdx.x, ip_tile_size)] = 0.0f;
    }
    __syncthreads();

    if (row >= 0 && row < num_rows && col >= 0 && col < num_cols) {

        if (row >= 1 && row < (num_rows - 1) &&
        col >= 1 && col < (num_cols - 1)) {

            if(threadIdx.y > 0 && threadIdx.y < (ip_tile_size - 1u) &&
               threadIdx.x > 0 && threadIdx.x < (ip_tile_size - 1u)) {

                op[elemIndex(row, col, num_cols)] =
                    sm[elemIndex(threadIdx.y, threadIdx.x, ip_tile_size)] +
                        sm[elemIndex(threadIdx.y - 1, threadIdx.x, ip_tile_size)] +
                        sm[elemIndex(threadIdx.y + 1, threadIdx.x, ip_tile_size)] +
                        sm[elemIndex(threadIdx.y, threadIdx.x - 1, ip_tile_size)] +
                        sm[elemIndex(threadIdx.y, threadIdx.x + 1, ip_tile_size)];
            }
        }
        else{// Boundary conditions.
            op[elemIndex(row, col, num_cols)] =
                ip[elemIndex(row, col, num_cols)];
        }
    }
}

Eigen::MatrixXf sum5PointStencil(Eigen::MatrixXf const &ip_mat,
                                 bool use_shared_mem)
{
    auto const ip_host = EigenUtils::toVec(ip_mat);

    if (ip_mat.rows() < 3 || ip_mat.cols() < 3) {
        throw std::invalid_argument{"Number of columns and rows should be minimum 2.\n"};
    }

    // Eliminate first and last rows and columns.
    auto const num_rows_op = ip_mat.rows();
    auto const num_cols_op = ip_mat.cols();

    auto res_host = std::vector<float>(num_rows_op * num_cols_op);

    // Allocate device GPU memory.
    auto ip_dev = static_cast<float *>(nullptr);
    auto res_dev = static_cast<float *>(nullptr);

    checkError(cudaMalloc(reinterpret_cast<void **>(&ip_dev), ip_host.size() * sizeof(float)),
               "allocation of device buffer for input vector");
    checkError(cudaMalloc(reinterpret_cast<void **>(&res_dev), res_host.size() * sizeof(float)),
               "allocation of device buffer for result vector");

    // Transfer data from the host to the device.
    checkError(cudaMemcpy(ip_dev, ip_host.data(),
                          ip_host.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "transfer of data from the input vector to the device");

    if (use_shared_mem) {

        auto const ip_tile_size = 16u;
        auto const op_tile_size = ip_tile_size - 2u;
        auto const block_size = dim3{ip_tile_size, ip_tile_size};
        auto const num_block_x = (static_cast<unsigned>(ip_mat.cols()) + op_tile_size - 1u) /
            op_tile_size;
        auto const num_block_y = (static_cast<unsigned>(ip_mat.rows()) + op_tile_size - 1u) /
            op_tile_size;
        auto const grid_size = dim3{num_block_x, num_block_y};
        auto const shared_mem_size = ip_tile_size * ip_tile_size * sizeof(float);

        sum_stencil_kern_sm<<<grid_size, block_size, shared_mem_size>>>(
            ip_dev, res_dev,
            static_cast<unsigned>(ip_mat.rows()),
            static_cast<unsigned>(ip_mat.cols()));
    }
    else {
        auto const block_size = dim3{16u, 16u};
        auto const num_block_x = (static_cast<unsigned>(ip_mat.cols()) + block_size.x - 1u) /
            block_size.x;
        auto const num_block_y = (static_cast<unsigned>(ip_mat.rows()) + block_size.y - 1u) /
            block_size.y;
        auto const grid_size = dim3{num_block_x, num_block_y};

        sum_stencil_kern<<<grid_size, block_size>>>(ip_dev, res_dev,
                                                    static_cast<unsigned>(ip_mat.rows()),
                                                    static_cast<unsigned>(ip_mat.cols()));
    }


    // Transfer result data from the device to host.
    checkError(cudaMemcpy(res_host.data(), res_dev,
                          res_host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "transfer results from the device to host");

    cudaFree(ip_dev);
    cudaFree(res_dev);

    return EigenUtils::toMat<float>(res_host, num_rows_op, num_cols_op);
}

}// Numeric::CUDA namespace.
