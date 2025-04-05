#ifndef PINNED_VECTOR_CUH
#define PINNED_VECTOR_CUH

#include <vector>

template <typename T>
class PinnedHostVector
{
private:
    T *buff_ = nullptr;
    unsigned num_elems_;

public:
    explicit PinnedHostVector(unsigned num_elems) : num_elems_(num_elems)
    {
        // Allocate pinned memory on the host.
        checkError(cudaHostAlloc(reinterpret_cast<void **>(&buff_), num_elems_ * sizeof(T), cudaHostAllocDefault),
                   "Allocation of pinned host memory");
    }
    ~PinnedHostVector()
    {
        // Free pinned memory on the host.
        checkError(cudaFreeHost(buff_), "Deallocation of pinned host memory");
    }

    T *data() noexcept
    {
        return buff_;
    }
    T const *data() const noexcept
    {
        return buff_;
    }
    size_t size() const noexcept
    {
        return num_elems_;
    }
};

#endif // PINNED_VECTOR_CUH
