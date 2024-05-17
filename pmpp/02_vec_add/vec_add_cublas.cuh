#ifndef VEC_ADD_CUBLAS_CUH
#define VEC_ADD_CUBLAS_CUH

#include "vec_add_impl_strategy.cuh"
#include "cublas_v2.h"

namespace PMPP
{
template<typename T>
class VecAddCublas : public VecAddImplStrategy<T>
{
protected:
	void launchKernel(T const* first_dev, T const* sec_dev, T* res_dev, unsigned num_elements) override;
};

template<typename T>
void VecAddCublas<T>::launchKernel(T const* first_dev, T const* sec_dev,
                                   T* res_dev, unsigned num_elements)
{
	// Since CUBLAS SAXPY is an in-place operation, 
	// copy the second vector to the result vector.
	cudaMemcpy(res_dev, sec_dev, num_elements * sizeof(T), cudaMemcpyDeviceToDevice);
	auto handle = cublasHandle_t{};
	cublasCreate(&handle);

	auto const alpha = static_cast<T>(1);
	cublasSaxpy(handle, static_cast<int>(num_elements), &alpha, first_dev, 1, res_dev, 1);
	cublasDestroy(handle);
}
} // namespace PMPP.
#endif


