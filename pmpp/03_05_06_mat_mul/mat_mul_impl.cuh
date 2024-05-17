#ifndef MAT_MUL_IMPL_CUH
#define MAT_MUL_IMPL_CUH

#include <Eigen/Core>
#include "../utils/dev_timer.cuh"

namespace PMPP::CUDA
{
template<typename T>
class MatMulImpl
{
public:
    virtual ~MatMulImpl() = default;

    virtual void launchKernel(T const *first, T const *sec, T *res,
                              unsigned num_rows_first,
                              unsigned num_cols_first,
                              unsigned num_cols_sec) = 0;
};

}// PMPP::CUDA namespace.

#endif //MAT_MUL_IMPL_CUH

