#ifndef VEC_ADD_IMPL_STRATEGY_CUH
#define VEC_ADD_IMPL_STRATEGY_CUH

#include "../../utils/dev_timer.cuh"
#include  <cuda_profiler_api.h>

namespace PMPP
{
template<typename T>
class VecAddImplStrategy
{
public:
    virtual ~VecAddImplStrategy() = default;
    VecAddImplStrategy() = default;
    virtual void launchKernel(T const *first_dev, T const *sec_dev, T *res_dev,
                              unsigned num_elements) = 0;
};
}// namespace PMPP.
#endif //VEC_ADD_IMPL_STRATEGY_CUH


