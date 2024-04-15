#include "dev_vector.cuh"
#include "check_error.cuh"

DevVector::DevVector(unsigned num_elems)
    :
    num_elems_{num_elems}
{
    checkError(cudaMalloc(reinterpret_cast<void **>(&buff_), num_elems_ * sizeof(float)),
               "allocation of device buffer for vector");
}
DevVector::~DevVector()
{
    cudaFree(buff_);
}

DevVector::DevVector(unsigned num_elems, float val)
    : DevVector{num_elems}
{
    checkError(cudaMemset(buff_, val, num_elems_ * sizeof(float)),
               "initialization of vector buffer");
}
DevVector::DevVector(std::vector<float> const &host)
    : DevVector{static_cast<unsigned>(std::size(host))}
{
    copyToDevice(*this, host);
}

std::vector<float> DevVector::hostCopy() const
{
    auto host = std::vector<float>(num_elems_);
    copyToHost(host, *this);
    return host;
}

void copyToDevice(DevVector &dev, std::vector<float> const &host)
{
    checkError(cudaMemcpy(dev.data(), host.data(), dev.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "transfer of data from the host to the device");
}

void copyToHost(std::vector<float> &host, DevVector const &dev)
{
    checkError(cudaMemcpy(host.data(), dev.data(), dev.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "transfer of data from the device to the host");
}