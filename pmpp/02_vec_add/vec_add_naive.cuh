#ifndef VECTOR_ADD_NAIVE_CUH
#define VECTOR_ADD_NAIVE_CUH

#include "vec_add_impl_strategy.cuh"
#include "../../utils/exec_config.cuh"
#include "../../utils/check_error.cuh"
#include <concepts>

namespace PMPP
{
template<typename T>
class VecAddNaive: public VecAddImplStrategy<T>
{
public:
    VecAddNaive() = default;
    virtual ~VecAddNaive() = default;

protected:
    void launchKernel(T const *first_dev, T const *sec_dev, T *res_dev,
                      unsigned num_elements) override;
};

template<typename T>
__global__ void vecAddKernel(T const *first, T const *sec, T *res, unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        res[idx] = first[idx] + sec[idx];
    }
}

template<typename T>
void
VecAddNaive<T>::launchKernel(T const *first_dev, T const *sec_dev,
                             T *res_dev, unsigned num_elements)
{
    auto const exec_params = ExecConfig::getParams(num_elements, vecAddKernel<T>, 0);
    vecAddKernel <<<exec_params.grid_dim, exec_params.block_dim>>>(
        first_dev, sec_dev, res_dev, num_elements);
    checkErrorKernel("vecAddKernel", true);
}

}// PMPP namespace.
#endif // !VECTOR_ADD_NAIVE_CUH