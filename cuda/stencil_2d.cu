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
                                 unsigned num_rows_ip, unsigned num_cols_ip)
{
    auto const row_ip = blockIdx.y * blockDim.y + threadIdx.y;
    auto const col_ip = blockIdx.x * blockDim.x + threadIdx.x;

    auto const num_cols_op = num_cols_ip - 2u;

    if ((row_ip > 0) && (row_ip < (num_rows_ip - 1u)) &&
        (col_ip > 0) && (col_ip < (num_cols_ip - 1u))) {

        auto const row_op = row_ip - 1u;
        auto const col_op = col_ip - 1u;

        op[elemIndex(row_op, col_op, num_cols_op)] =
            ip[elemIndex(row_ip, col_ip, num_cols_ip)] +
                ip[elemIndex(row_ip - 1u, col_ip, num_cols_ip)] +
                ip[elemIndex(row_ip + 1u, col_ip, num_cols_ip)] +
                ip[elemIndex(row_ip, col_ip - 1u, num_cols_ip)] +
                ip[elemIndex(row_ip, col_ip + 1u, num_cols_ip)];

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
    auto const num_rows_op = ip_mat.rows() - 2;
    auto const num_cols_op = ip_mat.cols() - 2;

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


    auto const block_size = dim3{16u, 16u};
    auto const num_block_x = (static_cast<unsigned>(ip_mat.cols()) + block_size.x - 1u) /
        block_size.x;
    auto const num_block_y = (static_cast<unsigned>(ip_mat.rows()) + block_size.y - 1u) /
        block_size.y;
    auto const grid_size = dim3{num_block_x, num_block_y};


    sum_stencil_kern<<<grid_size, block_size>>>(ip_dev, res_dev,
                                                static_cast<unsigned>(ip_mat.rows()),
                                                static_cast<unsigned>(ip_mat.cols()));

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
