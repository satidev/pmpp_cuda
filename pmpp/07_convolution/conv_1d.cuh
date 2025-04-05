#ifndef CONV_1D_CUH
#define CONV_1D_CUH

#include <vector>

namespace PMPP::CUDA
{
    std::vector<float> conv1D(std::vector<float> const &data,
                              std::vector<float> const &filter,
                              bool use_const_mem = false,
                              bool use_shared_mem = false);

} // Numeric::CUDA namespace.

#endif // CONV_1D_CUH
