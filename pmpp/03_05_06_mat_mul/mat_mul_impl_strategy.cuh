#ifndef MAT_MUL_IMPL_STRATEGY_CUH
#define MAT_MUL_IMPL_STRATEGY_CUH

#include "../../utils/dev_timer.cuh"

namespace PMPP
{
    template <typename T>
    class MatMulImplStrategy
    {
    public:
        virtual ~MatMulImplStrategy() = default;

        virtual void launchKernel(T const *first, T const *sec, T *res,
                                  unsigned num_rows_first,
                                  unsigned num_cols_first,
                                  unsigned num_cols_sec) = 0;
    };

} // PMPP namespace.

#endif // MAT_MUL_IMPL_STRATEGY_CUH
