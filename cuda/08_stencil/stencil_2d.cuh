#ifndef STENCIL_2D_CUH
#define STENCIL_2D_CUH

#include "Eigen/Core"

namespace Numeric::CUDA
{
// Sum of pixel and 4 nearest neighbors.
Eigen::MatrixXf sum5PointStencil(Eigen::MatrixXf const &ip_mat,
                                  bool use_shared_mem = false);
}// Numeric::CUDA namespace.

#endif //STENCIL_2D_CUH


