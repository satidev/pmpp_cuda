#ifndef DEV_VECTOR_ASYNC_CUH
#define DEV_VECTOR_ASYNC_CUH

#include <memory>
#include "stream_adaptor.cuh"
#include <iostream>

template<typename T>
class DevVectorAsync
{
public:
    explicit DevVectorAsync(std::shared_ptr<StreamAdaptor> stream,
                            unsigned num_elems);
    explicit DevVectorAsync(std::shared_ptr<StreamAdaptor> stream,
                            unsigned num_elems, T val);

    unsigned size() const noexcept
    {
        return num_elems_;
    }
    T *data()
    {
        return buff_;
    }
    T *data() const
    {
        return buff_;
    }

    ~DevVectorAsync();

    cudaStream_t stream() const
    {
        return stream_->getStream();
    }

private:
    std::shared_ptr<StreamAdaptor> stream_;
    unsigned num_elems_;
    T *buff_ = nullptr;
};

template<typename T>
DevVectorAsync<T>::~DevVectorAsync()
{
    auto const err = cudaFreeAsync(buff_, this->stream());
    if(err != cudaSuccess) {
        std::cerr << "Error freeing device buffer: " << cudaGetErrorString(err) << std::endl;
    }
}

template<typename T>
DevVectorAsync<T>::DevVectorAsync(std::shared_ptr<StreamAdaptor> stream,
                                  unsigned int num_elems,
                                  T val):
    DevVectorAsync{std::move(stream), num_elems}
{
    checkError(cudaMemsetAsync(buff_, val, num_elems_ * sizeof(T), this->stream()),
               "initialization of vector buffer");
}
template<typename T>
DevVectorAsync<T>::DevVectorAsync(std::shared_ptr<StreamAdaptor> stream, unsigned int num_elems)
    :
    stream_{std::move(stream)},
    num_elems_{num_elems}
{
    checkError(cudaMallocAsync(reinterpret_cast<void **>(&buff_), num_elems_ * sizeof(T),
                               this->stream()),
               "allocation of device buffer for vector");
}

#endif //DEV_VECTOR_ASYNC_CUH


