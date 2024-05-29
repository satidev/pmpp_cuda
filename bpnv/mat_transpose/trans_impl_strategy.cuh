#ifndef TRANS_IMPL_STRATEGY_CUH
#define TRANS_IMPL_STRATEGY_CUH

namespace BPNV
{
template<typename T>
class TransImplStrategy
{
public:
    virtual ~TransImplStrategy() = default;
    virtual void launchKernel(T const *input, T *output,
                              unsigned num_rows_input, unsigned num_cols_input) const = 0;
};

}// BPNV namespace.

#endif //TRANS_IMPL_STRATEGY_CUH


