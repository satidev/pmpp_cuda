#ifndef CONV_CUH
#define CONV_CUH

#include <vector>

namespace Numeric::CUDA
{
std::vector<float> conv1D(std::vector<float> const &data,
                          std::vector<float> const &filter,
                          bool use_const_mem = false,
                          bool use_shared_mem = false);

} // Numeric::CUDA namespace.

#endif //CONV_CUH
