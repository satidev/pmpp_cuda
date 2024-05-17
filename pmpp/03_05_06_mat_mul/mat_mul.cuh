#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include <memory>
#include "mat_mul_impl.cuh"
#include "../../utils/dev_config.cuh"
#include "../../utils/dev_vector.cuh"
#include <stdexcept>
#include "../../utils/dev_timer.cuh"
#include <tuple>
#include "../../utils/perf.cuh"
#include "../../utils/dev_vector_factory.cuh"
#include <dlib/matrix.h>

namespace PMPP
{
template<typename T>
class MatMul
{
private:
    std::unique_ptr<MatMulImpl<T>> impl_;

public:
    explicit MatMul(std::unique_ptr<MatMulImpl<T>> impl)
        :
        impl_{std::move(impl)}
    {}

    std::tuple<dlib::matrix<T>, PerfInfo> run(dlib::matrix<T> const &first,
                                              dlib::matrix<T> const &sec) const;
};

template<typename T>
std::tuple<dlib::matrix<T>, PerfInfo> MatMul<T>::run(dlib::matrix<T> const &first,
                                                     dlib::matrix<T> const &sec) const
{
    if (first.nc() != sec.nr()) {
        throw std::invalid_argument{"Invalid size for matrix multiplication."};
    }

    // Check the device has enough global memory to store all vectors.
    auto const &dev_config = DeviceConfigSingleton::getInstance();
    if ((first.nr() * first.nc() +
        sec.nr() * sec.nc() +
        first.nr() * sec.nc()) * sizeof(T) >
        dev_config.getDevProps(0).global_mem_size) {
        throw std::runtime_error{"Insufficient global memory on the device."};
    }

    // Allocate device memory and copy the matrix data.
    auto const first_dev = DevVectorFactory::create(
        std::vector<T>(first.begin(), first.end()));
    auto const sec_dev = DevVectorFactory::create(
        std::vector<T>(first.begin(), first.end()));

    auto const num_rows_first = static_cast<unsigned>(first.nr());
    auto const num_cols_first = static_cast<unsigned>(first.nc());
    auto const num_cols_sec = static_cast<unsigned>(sec.nc());

    // Allocate device memory for the result.
    auto res_dev = DevVector<T>{num_rows_first * num_cols_sec};

    auto timer = DevTimer{};
    timer.tic();
    impl_->launchKernel(first_dev.data(), sec_dev.data(), res_dev.data(),
                        num_rows_first, num_cols_first, num_cols_sec);
    auto const time_taken = timer.toc();

    return std::make_tuple(dlib::mat(HostDevCopy::hostCopy(res_dev).data(),
                                     first.nr(),
                                     sec.nc()),
                           PerfInfo{time_taken});
}

void matMulPerfTest();
}// PMPP namespace.

#endif //MAT_MUL_CUH
