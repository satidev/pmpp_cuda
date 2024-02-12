#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include "../cpp/eigen_utils.h"
#include "../thirdparty/Eigen/Core"

namespace Numeric::CUDA
{
Eigen::MatrixXf matMul(Eigen::MatrixXf const &a, Eigen::MatrixXf const &b);
}// Numeric namespace.

#endif //MAT_MUL_CUH
