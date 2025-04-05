#include <chrono>
#include "mem_bandwidth.cuh"
#include "../../utils/check_error.cuh"
#include "../../utils/dev_timer.cuh"
#include <iostream>
#include <vector>
#include "../../utils/perf.cuh"
#include "../../utils/dev_vector.cuh"
#include "../../utils/pinned_host_vector.cuh"
#include "../../utils/host_dev_copy.cuh"

namespace BPNV::MemoryBandwidth
{
    float pageableMem(unsigned num_elems, unsigned num_transfers)
    {
        auto const data_host = std::vector<float>(num_elems, 1.0f);

        // Allocate pageable device memory.
        auto data_dev = DevVector<float>{num_elems};

        // Copy data to device memory.
        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();
        for (auto i = 0u; i < num_transfers; ++i)
        {
            HostDevCopy::copyFromHostToDevice(data_dev, data_host);
        }
        cudaDeviceSynchronize();

        auto const duration = timer.toc();
        auto const num_bytes = num_elems * sizeof(float) * num_transfers;
        return computeBandwidth(num_bytes, duration);
    }

    float pinnedMem(unsigned num_elems, unsigned num_transfers)
    {
        auto data_host = PinnedHostVector<float>{num_elems};
        std::fill_n(data_host.data(), num_elems, 1.0f);

        auto const stream = StreamAdaptor{};
        auto data_dev = DevVectorAsync<float>{stream, num_elems};

        // Copy data to device memory.
        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();
        for (auto i = 0u; i < num_transfers; ++i)
        {
            HostDevCopy::copyFromHostToDevice(data_dev, data_host);
        }
        cudaDeviceSynchronize();

        auto const duration = timer.toc();
        auto const num_bytes = num_elems * sizeof(float) * num_transfers;
        return computeBandwidth(num_bytes, duration);
    }

    PerfTestResult runPerfTest(unsigned num_rep)
    {
        std::cout << "Memory bandwidth test: start" << std::endl;

        auto const num_elems = 1 << 24;
        auto constexpr num_transfers = 10u;
        std::cout << "Transfer data size: " << num_elems * sizeof(float) << " bytes" << std::endl;

        auto perf_info = PerfTestResult{};
        for (auto run_idx = 0u; run_idx < num_rep; ++run_idx)
        {
            perf_info["pageable-sync"].emplace_back(pageableMem(num_elems, num_transfers));
            perf_info["pinned-async"].emplace_back(pinnedMem(num_elems, num_transfers));
        }
        std::cout << "Memory bandwidth test: end" << std::endl;

        return perf_info;
    }

} // namespace MemoryBandwidth
