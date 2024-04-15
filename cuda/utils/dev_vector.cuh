#ifndef DEV_VECTOR_CUH
#define DEV_VECTOR_CUH

#include <vector>

class DevVector
{
public:
    explicit DevVector(unsigned num_elems);
    explicit DevVector(unsigned num_elems, float val);
    explicit DevVector(std::vector<float> const &host);

    unsigned size() const noexcept
    {
        return num_elems_;
    }
    float *data()
    {
        return buff_;
    }
    float *data() const
    {
        return buff_;
    }

    std::vector<float> hostCopy() const;

    ~DevVector();

private:
    unsigned num_elems_;
    float *buff_ = nullptr;
};

// copy data from host to device.
void copyToDevice(DevVector &dev, std::vector<float> const &host);

// copy data from device to host.
void copyToHost(std::vector<float> &host, DevVector const &dev);

#endif //DEV_VECTOR_CUH


