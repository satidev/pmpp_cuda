#ifndef PINNED_VECTOR_CUH
#define PINNED_VECTOR_CUH

#include <vector>

template<typename T>
class PinnedVector
{
private:
    std::vector<T> vec_;

public:
    explicit PinnedVector(std::vector<T> &&vec)
        :
        vec_{std::move(vec)}
    {
        checkError(cudaHostRegister((void *) vec_.data(),
                                    vec_.size() * sizeof(T),
                                    cudaHostRegisterDefault),
                   "registering pinned memory");
    }
    ~PinnedVector()
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
};

#endif //PINNED_VECTOR_CUH


