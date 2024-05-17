#ifndef DEV_VECTOR_FACTORY_CUH
#define DEV_VECTOR_FACTORY_CUH

#include <vector>
#include "dev_vector.cuh"
#include "host_dev_copy.cuh"

namespace DevVectorFactory
{
template<typename T>
DevVector<T> create(std::vector<T> const &host);

template<typename T>
DevVector<T> create(std::vector<T> const &host)
{
    auto const num_elems = static_cast<unsigned>(std::size(host));
    auto dev =  DevVector<T>{num_elems};
    HostDevCopy::copyToDevice(dev, host);
    return dev;
}

}// DevVectorFactory namespace.
#endif //DEV_VECTOR_FACTORY_CUH


