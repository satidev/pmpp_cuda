#ifndef MAPPED_VECTOR_CUH
#define MAPPED_VECTOR_CUH

#include <vector>
#include "check_error.cuh"
#include <stdexcept>

template<typename T>
class MappedVector
{
private:
    std::vector<T> vec_;
    T *mapped_ptr_ = nullptr;

public:
    explicit MappedVector(std::vector<T> &&vec)
        :
        vec_{std::move(vec)}
    {
        auto dev_prop = cudaDeviceProp{};
        checkError(cudaGetDeviceProperties(&dev_prop, 0), "getting device properties");
        if (!dev_prop.canMapHostMemory) {
            throw std::runtime_error{"Zero copy memory is not supported"};
        }
        checkError(cudaSetDeviceFlags(cudaDeviceMapHost),
                   "setting device flags for zero copy memory");
        checkError(cudaHostRegister((void *) vec_.data(), vec_.size() * sizeof(float),
                                    cudaHostRegisterMapped),
                   "registering input data host memory");
        checkError(cudaHostGetDevicePointer((void **) &mapped_ptr_,
                                            (void *) vec_.data(), 0),
                   "getting device pointer for mapped memory");
    }

    ~MappedVector()
    {
        cudaHostUnregister((void *) vec_.data());
    }

    T *data() noexcept
    {
        return vec_.data();
    }
    T const *data() const noexcept
    {
        return vec_.data();
    }
    size_t size() const noexcept
    {
        return vec_.size();
    }
    T *mappedData() noexcept
    {
        return mapped_ptr_;
    }
    T const *mappedData() const noexcept
    {
        return mapped_ptr_;
    }
};

#endif //MAPPED_VECTOR_CUH


