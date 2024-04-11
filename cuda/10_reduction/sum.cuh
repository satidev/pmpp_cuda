#ifndef SUM_CUH
#define SUM_CUH

#include <vector>

namespace Numeric::CUDA
{
    float sumSeq(std::vector<float> const &data);
    float sumParallel(std::vector<float> const &data_host);

}// Numeric::CUDA namespace.

#endif //SUM_CUH

