#ifndef STENCIL_1D_CUH
#define STENCIL_1D_CUH

#include <vector>

namespace Numeric::CUDA
{
    // Finite difference, 3-point stencil pattern.
    std::vector<float> diff(std::vector<float> const &ip_vec);
}// Numeric::CUDA namespace.

#endif //STENCIL_1D_CUH


