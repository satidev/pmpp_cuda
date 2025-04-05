#ifndef HOST_DEV_COPY_CUH
#define HOST_DEV_COPY_CUH

#include "dev_vector.cuh"
#include "dev_vector_async.cuh"
#include "pinned_host_vector.cuh"
#include <vector>

namespace HostDevCopy
{
    template <typename T>
    void copyFromHostToDevice(DevVector<T> &dst, std::vector<T> const &src);

    template <typename T>
    void copyFromHostToDevice(DevVectorAsync<T> &dst, PinnedHostVector<T> const &src);

    template <typename T>
    void copyFromDeviceToHost(std::vector<T> &dst, DevVector<T> const &src);

    template <typename T>
    void copyFromDeviceToHost(PinnedHostVector<T> &dst, DevVectorAsync<T> const &src);

    template <typename T>
    std::vector<T> hostCopy(DevVector<T> const &src);

    template <typename T>
    void copyFromHostToDevice(DevVectorAsync<T> &dst, PinnedHostVector<T> const &src)
    {
        if (std::size(src) != dst.size())
        {
            throw std::runtime_error("Size mismatch between pinned and device vectors");
        }
        checkError(cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, dst.stream()),
                   "transfer of data from the host to the device");
    }

    template <typename T>
    void copyFromDeviceToHost(PinnedHostVector<T> &dst, DevVectorAsync<T> const &src)
    {
        if (std::size(dst) != src.size())
        {
            throw std::runtime_error("Size mismatch between pinned and device vectors");
        }
        checkError(cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(T),
                                   cudaMemcpyDeviceToHost, src.stream()),
                   "transfer of data from the device to the host");
    }

    // copy data from host to device.
    template <typename T>
    void copyFromHostToDevice(DevVector<T> &dst, std::vector<T> const &src)
    {
        checkError(cudaMemcpy(dst.data(), src.data(), dst.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "transfer of data from the host to the device");
    }

    // copy data from device to host.
    template <typename T>
    void copyFromDeviceToHost(std::vector<T> &dst, DevVector<T> const &src)
    {
        checkError(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "transfer of data from the device to the host");
    }

    template <typename T>
    std::vector<T> hostCopy(DevVector<T> const &src)
    {
        auto dst = std::vector<T>(std::size(src));
        copyFromDeviceToHost(dst, src);
        return dst;
    }

} // HostDevCopy namespace.

#endif // HOST_DEV_COPY_CUH
