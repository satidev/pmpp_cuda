#ifndef SUM_CUH
#define SUM_CUH

#include <vector>

namespace PMPP::CUDA
{
    float sumSeq(std::vector<float> const &data);
    enum class ReductionStrategy {
        NAIVE,
        SIMPLE,
        SIMPLE_MIN_DIV,
        SIMPLE_MIN_DIV_SHARED,
        SIMPLE_MIN_DIV_SHARED_MULT_BLOCKS,
        SIMPLE_MIN_DIV_SHARED_MULT_BLOCKS_COARSE
    };
    float sumParallel(std::vector<float> const &data_host,
                      ReductionStrategy strategy = ReductionStrategy::SIMPLE);

    namespace Detail
    {
        std::pair<unsigned, unsigned> execConfig(unsigned num_data_elems,
                                                 ReductionStrategy strategy);
    }

}// Numeric::CUDA namespace.

#endif //SUM_CUH

