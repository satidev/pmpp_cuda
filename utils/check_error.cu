#include "check_error.cuh"
#include <stdexcept>

void checkError(cudaError_t result, std::string const &func_desc)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error{
            "CUDA runtime error during " + func_desc + ".\n" + cudaGetErrorString(result)};
    }
}
void checkErrorKernel(std::string const &kern_desc, bool block)
{
    // Checking synchronous errors like invalid execution configuration.
    auto const err_sync = cudaGetLastError();
    if (err_sync != cudaSuccess)
    {
        throw std::runtime_error{
            "CUDA kernel (synchronous) error during " + kern_desc + ".\n" + cudaGetErrorString(err_sync)};
    }

    if (block)
    {
        // Checking asynchronous errors like out-of-bound memory within the kernel.
        auto const err_async = cudaDeviceSynchronize();
        if (err_async != cudaSuccess)
        {
            throw std::runtime_error{
                "CUDA kernel (asynchronous) error during " + kern_desc + ".\n" + cudaGetErrorString(err_sync)};
        }
    }
}
