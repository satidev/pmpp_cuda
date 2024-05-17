#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include <dlib/matrix.h>
#include "trans_impl_strategy.cuh"
#include "../../utils/dev_vector_factory.cuh"
#include "../../utils/dev_timer.cuh"
#include "../../utils/perf.cuh"
#include <memory>
#include <tuple>
#include "../../utils/dev_config.cuh"

namespace BPNV
{
template<typename T>
class Transpose
{
private:
    std::unique_ptr<TransImplStrategy<T>> impl_;

public:
    explicit Transpose(std::unique_ptr<TransImplStrategy<T>> impl)
        :
        impl_{std::move(impl)}
    {}

    std::tuple<dlib::matrix<T>, PerfInfo> run(dlib::matrix<T> const &mat) const;
};

template<typename T>
std::tuple<dlib::matrix<T>, PerfInfo> Transpose<T>::run(dlib::matrix<T> const &mat) const
{
    // Check the device has enough global memory to store all vectors.
    auto const &dev_config = DeviceConfigSingleton::getInstance();
    if ((2 * mat.nr() * mat.nc() * sizeof(T)) > dev_config.getDevProps(0).global_mem_size) {
        throw std::runtime_error{"Insufficient global memory on the device."};
    }


    // Allocate device memory and copy the matrix data.
    auto const mat_dev = DevVectorFactory::create(
        std::vector<T>(mat.begin(), mat.end()));

    // Allocate device memory for the result.
    auto const num_rows_input = static_cast<unsigned>(mat.nr());
    auto const num_cols_input = static_cast<unsigned>(mat.nc());
    auto res_dev = DevVector<T>{num_rows_input * num_cols_input};

    // Launch the kernel.
    auto timer = DevTimer{};
    timer.tic();
    impl_->launchKernel(mat_dev.data(), res_dev.data(),
                        num_rows_input, num_cols_input);
    auto const time_taken_sec = timer.toc();

    // Copy the result back to the host.
    return std::make_tuple(dlib::mat(HostDevCopy::hostCopy(res_dev).data(),
                                     num_cols_input,
                                     num_rows_input),
                           PerfInfo{time_taken_sec});
}
void transposePerfTest();
}// BPNV namespace.

#endif //TRANSPOSE_CUH
