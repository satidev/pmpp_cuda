#ifndef CONV_2D_CUH
#define CONV_2D_CUH

#include "Eigen/Core"

namespace PMPP::CUDA
{
Eigen::MatrixXf conv2D(Eigen::MatrixXf const &data,
                       Eigen::MatrixXf const &filter,
                       bool use_shared_mem = false);
}// Numeric::CUDA namespace.

#endif //CONV_2D_CUH


