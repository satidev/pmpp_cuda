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
    explicit DevVector(std::vector<T> const &host);

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

    std::vector<T> hostCopy() const;

    ~DevVector();

private:
    unsigned num_elems_;
    T *buff_ = nullptr;
};

template<typename T>
std::vector<T> DevVector<T>::hostCopy() const
{
    auto host = std::vector<T>(num_elems_);
    copyToHost(host, *this);
    return host;
}

template<typename T>
DevVector<T>::~DevVector()
{
    cudaFree(buff_);
}

template<typename T>
DevVector<T>::DevVector(const std::vector<T> &host)
    :
    DevVector{static_cast<unsigned>(std::size(host))}
{
    copyToDevice(*this, host);
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

// copy data from host to device.
template<typename T>
void copyToDevice(DevVector<T> &dev, std::vector<T> const &host)
{
    checkError(cudaMemcpy(dev.data(), host.data(), dev.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               "transfer of data from the host to the device");
}

// copy data from device to host.
template<typename T>
void copyToHost(std::vector<T> &host, DevVector<T> const &dev)
{
    checkError(cudaMemcpy(host.data(), dev.data(), dev.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "transfer of data from the device to the host");
}


#endif //DEV_VECTOR_CUH


