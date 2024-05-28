#ifndef DEV_VECTOR_CUH
#define DEV_VECTOR_CUH

#include <vector>
#include "check_error.cuh"

template<typename T>
class DevVector
{
public:
    explicit DevVector(unsigned num_elems);
    explicit DevVector(unsigned num_elems, T val);

    unsigned size() const noexcept
    {
        return num_elems_;
    }
    T *data()
    {
        return buff_;
    }
    T const *data() const
    {
        return buff_;
    }

    ~DevVector();

private:
    unsigned num_elems_;
    T *buff_ = nullptr;
};

template<typename T>
DevVector<T>::~DevVector()
{
    cudaFree(buff_);
}

template<typename T>
DevVector<T>::DevVector(unsigned num_elems, T val)
    :
    DevVector{num_elems}
{
    checkError(cudaMemset(buff_, val, num_elems_ * sizeof(T)),
               "initialization of vector buffer");
}

template<typename T>
DevVector<T>::DevVector(unsigned num_elems)
    :
    num_elems_{num_elems}
{
    checkError(cudaMalloc(reinterpret_cast<void **>(&buff_), num_elems_ * sizeof(T)),
               "allocation of device buffer for vector");
}

#endif //DEV_VECTOR_CUH


