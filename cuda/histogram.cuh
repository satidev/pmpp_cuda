#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <vector>

namespace Numeric::CUDA
{
std::vector<unsigned> histogram(std::vector<bool> const &data_host);
}// Numeric::CUDA namespace.

#endif //HISTOGRAM_CUH

