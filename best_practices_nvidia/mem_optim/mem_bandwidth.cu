#include <chrono>
#include "mem_bandwidth.cuh"
#include "../../utils/check_error.cuh"
#include "../../utils/dev_timer.cuh"
#include <iostream>
#include <vector>
#include "../../utils/perf.cuh"

namespace BPNV::MemoryBandwidth
{
float pageableMem(unsigned num_elems, unsigned num_transfers)
{
    // Allocate host memory.
    auto const data_host = std::vector<float>(num_elems, 1.0f);
    auto data_dev = static_cast<float *>(nullptr);

    // Allocate pageable device memory.
    auto const num_bytes = num_elems * sizeof(float);
    checkError(cudaMalloc(reinterpret_cast<void **>(&data_dev), num_bytes),
               "allocating pageable device memory");

    // Copy data to device memory.
    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();
    for (auto i = 0u; i < num_transfers; ++i) {
        checkError(cudaMemcpy(data_dev, data_host.data(), num_bytes, cudaMemcpyHostToDevice),
                   "copying data to device");
    }
    cudaDeviceSynchronize();
    auto const duration = timer.toc();
    cudaFree(data_dev);

    return computeBandwidth(num_bytes * num_transfers, MilliSeconds{duration});
}

float pinnedMem(unsigned num_elems, unsigned num_transfers)
{
    // Host data.
    auto const num_bytes = num_elems * sizeof(float);
    auto const data_host = std::vector<float>(num_elems, 1.0f);

    // Register host memory.
    checkError(cudaHostRegister((void *) (data_host.data()), num_bytes, cudaHostRegisterDefault),
               "register host memory");

    // Allocate device memory.
    auto data_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&data_dev), num_bytes),
               "allocating device memory");

    // Copy data to device memory.
    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();
    for (auto i = 0u; i < num_transfers; ++i) {
        checkError(cudaMemcpyAsync(data_dev, data_host.data(), num_bytes, cudaMemcpyHostToDevice),
                   "copying data to device");
    }
    cudaDeviceSynchronize();
    auto const duration = timer.toc();

    checkError(cudaHostUnregister((void *) data_host.data()), "unregister host memory");
    cudaFree(data_dev);

    return computeBandwidth(num_bytes * num_transfers, MilliSeconds{duration});
}

PerfTestResult runPerfTest(unsigned num_rep)
{
    std::cout << "Memory bandwidth test: start" << std::endl;

    auto const num_elems = 1 << 24;
    auto constexpr num_transfers = 10u;
    std::cout << "Data size: " << num_elems * sizeof(float) << " bytes" << std::endl;

    auto perf_info = PerfTestResult{};
    for (auto run_idx = 0u; run_idx < num_rep; ++run_idx) {
        perf_info["pageable-sync"].emplace_back(pageableMem(num_elems, num_transfers));
        perf_info["pinned-async"].emplace_back(pinnedMem(num_elems, num_transfers));
    }
    std::cout << "Memory bandwidth test: end" << std::endl;

    return perf_info;
}

}// namespace MemoryBandwidth
