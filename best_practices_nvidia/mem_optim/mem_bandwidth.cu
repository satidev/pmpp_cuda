#include <chrono>
#include "mem_bandwidth.cuh"
#include "../../utils/check_error.cuh"
#include "../../utils/dev_timer.cuh"
#include <iostream>
#include <vector>

namespace MemoryBandwidth
{
float pageableMem(unsigned num_elems, unsigned num_reps)
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
    for (auto i = 0u; i < num_reps; ++i) {
        checkError(cudaMemcpy(data_dev, data_host.data(), num_bytes, cudaMemcpyHostToDevice),
                   "copying data to device");
    }
    cudaDeviceSynchronize();
    auto const duration = timer.toc();
    cudaFree(data_dev);

    auto const num_bytes_total = num_bytes * num_reps / (1 << 30);
    auto const bandwidth = static_cast<float>(num_bytes_total) / duration;

    return bandwidth;
}

float pinnedMem(unsigned num_elems, unsigned num_reps)
{
    // Allocate pinned host memory.
    auto const num_bytes = num_elems * sizeof(float);
    auto data_host = static_cast<float *>(nullptr);
    checkError(cudaMallocHost(reinterpret_cast<void **>(&data_host), num_bytes),
               "allocating pinned host memory");
    // Initialize pinned memory.
    for (auto i = 0u; i < num_elems; ++i) {
        data_host[i] = 1.0f;
    }

    // Allocate device memory.
    auto data_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&data_dev), num_bytes),
               "allocating device memory");

    // Copy data to device memory.
    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();
    for (auto i = 0u; i < num_reps; ++i) {
        checkError(cudaMemcpyAsync(data_dev, data_host, num_bytes, cudaMemcpyHostToDevice),
                   "copying data to device");
    }
    cudaDeviceSynchronize();
    auto const duration = timer.toc();

    cudaFreeHost(data_host);
    cudaFree(data_dev);

    auto const num_bytes_total = num_bytes * num_reps / (1 << 30);
    auto const bandwidth = static_cast<float>(num_bytes_total) / duration;

    return bandwidth;
}

float pinnedMemRegister(unsigned num_elems, unsigned num_reps)
{
    // Host data.
    auto const num_bytes = num_elems * sizeof(float);
    auto const data_host = std::vector<float>(num_elems, 1.0f);

    // Register host memory.
    checkError(cudaHostRegister((void *)(data_host.data()), num_bytes, cudaHostRegisterDefault),
               "register host memory");

    // Allocate device memory.
    auto data_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&data_dev), num_bytes),
               "allocating device memory");

    // Copy data to device memory.
    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();
    for (auto i = 0u; i < num_reps; ++i) {
        checkError(cudaMemcpyAsync(data_dev, data_host.data(), num_bytes, cudaMemcpyHostToDevice),
                   "copying data to device");
    }
    cudaDeviceSynchronize();
    auto const duration = timer.toc();

    checkError(cudaHostUnregister((void *) data_host.data()), "unregister host memory");
    cudaFree(data_dev);

    auto const num_bytes_total = num_bytes * num_reps / (1 << 30);
    auto const bandwidth = static_cast<float>(num_bytes_total) / duration;

    return bandwidth;
}

void runPerfTest()
{
    auto const num_elems = 1 << 24;
    auto const num_reps = 100;
    auto const bandwidth_pageable = pageableMem(num_elems, num_reps);
    std::cout << "Pageable memory bandwidth: " << bandwidth_pageable << " GB/s\n";

    auto const bandwidth_pinned = pinnedMem(num_elems, num_reps);
    std::cout << "Pinned memory bandwidth: " << bandwidth_pinned << " GB/s\n";

    auto const bandwidth_pinned_reg = pinnedMemRegister(num_elems, num_reps);
    std::cout << "Pinned memory (reg) bandwidth: " << bandwidth_pinned_reg << " GB/s\n";
}
}// namespace MemoryBandwidth
