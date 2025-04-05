#ifndef STENCIL_1D_CUH
#define STENCIL_1D_CUH

#include <vector>

namespace PMPP::CUDA
{
    // Finite difference, 3-point stencil pattern.
    std::vector<float> diff(std::vector<float> const &ip_vec);

    // Sum of 3 nearest neighbors.
    std::vector<float> sum3Point(std::vector<float> const &ip_vec,
                                 bool use_shared_mem = false);
} // Numeric::CUDA namespace.

#endif // STENCIL_1D_CUH
