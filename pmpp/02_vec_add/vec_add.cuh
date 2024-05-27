#ifndef VEC_ADD_CUH
#define VEC_ADD_CUH

#include <vector>
#include "vec_add_impl_strategy.cuh"
#include <memory>
#include <stdexcept>
#include "../../utils/dev_vector.cuh"
#include "../../utils/dev_vector_factory.cuh"
#include <tuple>
#include "../../utils/perf.cuh"

namespace PMPP
{
template<typename T>
class VectorAdd
{
private:
    std::unique_ptr<VecAddImplStrategy<T>> impl_;

public:
    explicit VectorAdd(std::unique_ptr<VecAddImplStrategy<T>> impl)
        : impl_{std::move(impl)}
    {}
    std::tuple<std::vector<T>, kernelPerfInfo> run(std::vector<T> const &first,
                                                   std::vector<T> const &sec) const;
};

template<typename T>
std::tuple<std::vector<T>, kernelPerfInfo> VectorAdd<T>::run(std::vector<T> const &first,
                                                             std::vector<T> const &sec) const
{
    if (std::size(first) != std::size(sec)) {
        throw std::invalid_argument{"The vectors must have the same size"};
    }

    // Copy the vectors to the device.
    auto const first_dev = DevVectorFactory::create(first);
    auto const sec_dev = DevVectorFactory::create(sec);

    auto const num_elems = static_cast<unsigned>(std::size(first));
    auto res_dev = DevVector<T>{num_elems};
    auto timer = DevTimer{};
    timer.tic();
    impl_->launchKernel(first_dev.data(), sec_dev.data(),
                        res_dev.data(), num_elems);
    auto const time = timer.toc();
    return std::make_tuple(HostDevCopy::hostCopy(res_dev), kernelPerfInfo{time});
}

void vecAddPerfTest();
} // namespace PMPP.
#endif// !VEC_ADD_CUH


