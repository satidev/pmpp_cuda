#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include "../utils/eigen_utils.h"
#include "Eigen/Core"

namespace Numeric::CUDA
{
Eigen::MatrixXf matMul(Eigen::MatrixXf const &a,
                       Eigen::MatrixXf const &b,
                       bool use_shared_mem = false);
}// Numeric namespace.

#endif //MAT_MUL_CUH
