#ifndef HOST_DEV_COPY_CUH
#define HOST_DEV_COPY_CUH

#include <vector>
#include "dev_vector.cuh"

namespace HostDevCopy
{
template<typename T>
void copyToDevice(DevVector<T> &dst, std::vector<T> const &src);

template<typename T>
void copyToHost(std::vector<T> &dst, DevVector<T> const &src);

template<typename T>
std::vector<T> hostCopy(DevVector<T> const &src);

// copy data from host to device.
template<typename T>
void copyToDevice(DevVector<T> &dst, std::vector<T> const &src)
{
    checkError(cudaMemcpy(dst.data(), src.data(), dst.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               "transfer of data from the host to the device");
}

// copy data from device to host.
template<typename T>
void copyToHost(std::vector<T> &dst, DevVector<T> const &src)
{
    checkError(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "transfer of data from the device to the host");
}

template<typename T>
std::vector<T> hostCopy(DevVector<T> const &src)
{
    auto dst = std::vector<T>(std::size(src));
    copyToHost(dst, src);
    return dst;
}

}// HostDevCopy namespace.

#endif //HOST_DEV_COPY_CUH
