#ifndef VEC_ADD_CUH
#define VEC_ADD_CUH

#include <vector>

namespace Numeric::CUDA
{
std::vector<float> vecAdd(std::vector<float> const &first,
                          std::vector<float> const &sec);
}

#endif //VEC_ADD_CUH
