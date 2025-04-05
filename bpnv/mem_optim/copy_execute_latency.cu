#include "copy_execute_latency.cuh"
#include "../../utils/dev_timer.cuh"
#include "../../utils/check_error.cuh"
#include "../../utils/exec_config.cuh"
#include "../../utils/stream_adaptor.cuh"
#include "../../utils/dev_vector_async.cuh"
#include "../../utils/host_dev_copy.cuh"
#include "../../utils/pinned_host_vector.cuh"
#include "../../utils/mapped_vector.cuh"
#include <memory>

namespace BPNV::CopyExecuteLatency
{
    __global__ void sqKernel(float const *input, float *output,
                             unsigned num_elems)
    {
        auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elems)
        {
            output[idx] = input[idx] * input[idx];
        }
    }

    MilliSeconds seqCopyExecutePageable(unsigned num_elems)
    {
        // Allocate input data in host memory.
        auto constexpr init_val = 2.0f;
        auto const input_host = std::vector<float>(num_elems, init_val);

        // Allocate device memory for the input data.
        auto input_dev = DevVector<float>{num_elems};

        // Allocate memory for the result in the host and device.
        auto output_host = std::vector<float>(num_elems);
        auto output_dev = DevVector<float>{num_elems};

        auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);

        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();

        HostDevCopy::copyFromHostToDevice(input_dev, input_host);
        sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
            input_dev.data(), output_dev.data(), num_elems);
        HostDevCopy::copyFromDeviceToHost(output_host, output_dev);

        cudaDeviceSynchronize();
        auto const duration = timer.toc();

        if (!Detail::hasSameVal(output_host, init_val * init_val))
        {
            std::cerr << "Error: Kernel execution failed\n";
            std::exit(1);
        }

        return duration;
    }

    MilliSeconds seqCopyExecuteUnified(unsigned num_elems)
    {
        // Allocate input data in unified memory that can be
        // accessed by both the host and the device.
        auto constexpr init_val = 2.0f;
        auto const num_bytes = num_elems * sizeof(float);
        auto input_data = static_cast<float *>(nullptr);
        checkError(cudaMallocManaged(reinterpret_cast<void **>(&input_data), num_bytes),
                   "allocating unified memory for input data");
        for (auto i = 0u; i < num_elems; ++i)
        {
            input_data[i] = init_val;
        }

        // Allocate unified memory for the result.
        auto output_data = static_cast<float *>(nullptr);
        checkError(cudaMallocManaged(reinterpret_cast<void **>(&output_data), num_bytes),
                   "allocating device memory for output data");

        auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);
        // No explicit data transfer is required.
        // Execute the kernel.
        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();
        sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
            input_data, output_data, num_elems);

        cudaDeviceSynchronize();
        auto const duration = timer.toc();

        if (!Detail::hasSameVal(std::span<float>{output_data, num_elems}, init_val * init_val))
        {
            std::cerr << "Error: Kernel execution failed\n";
            std::exit(1);
        }

        // Clean up.
        cudaFree(input_data);
        cudaFree(output_data);

        return duration;
    }

    MilliSeconds seqCopyExecutePinned(unsigned num_elems)
    {
        // Allocate input data in host pinned memory.
        auto constexpr init_val = 2.0f;
        auto input_host = PinnedHostVector<float>{num_elems};
        std::fill_n(input_host.data(), num_elems, init_val);

        // Allocate device memory for the input data.
        auto stream = StreamAdaptor{};
        auto input_dev = DevVectorAsync<float>{stream, num_elems};

        // Allocate memory for the result in the host and device.
        auto output_host = PinnedHostVector<float>{num_elems};
        auto output_dev = DevVectorAsync<float>{stream, num_elems};

        auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);

        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();

        HostDevCopy::copyFromHostToDevice(input_dev, input_host);
        sqKernel<<<exec_params.grid_dim, exec_params.block_dim, 0, stream.getStream()>>>(
            input_dev.data(), output_dev.data(), num_elems);
        HostDevCopy::copyFromDeviceToHost(output_host, output_dev);

        cudaDeviceSynchronize();
        auto duration = timer.toc();

        // Check the results
        auto constexpr res_val = init_val * init_val;
        if (!Detail::hasSameVal(std::span<float>{output_host.data(),
                                                 output_host.size()},
                                res_val))
        {
            std::cerr << "Error: Kernel execution failed\n";
            std::exit(1);
        }

        return duration;
    }

    MilliSeconds stagedConcurrentCopyExecute(unsigned num_elems, unsigned num_streams)
    {
        // Allocate input data in host memory.
        auto constexpr init_val = 2.0f;
        auto input_host = PinnedHostVector<float>{num_elems};
        std::fill_n(input_host.data(), num_elems, init_val);

        // Allocate input data in device memory and transfer the data.
        auto const num_elem_stream = num_elems / num_streams;
        auto const num_byte_stream = num_elem_stream * sizeof(float);
        auto input_dev = static_cast<float *>(nullptr);
        checkError(cudaMalloc(reinterpret_cast<void **>(&input_dev), num_byte_stream),
                   "allocating device memory for input data");

        // Allocate device memory for the result.
        auto output_dev = static_cast<float *>(nullptr);
        checkError(cudaMalloc(reinterpret_cast<void **>(&output_dev), num_byte_stream),
                   "allocating device memory for output data");

        // Allocate the host memory for the result.
        auto res_host = PinnedHostVector<float>{num_elems};

        // Allocate streams.
        auto streams = std::vector<StreamAdaptor>(num_streams);

        auto const exec_params = ExecConfig::getParams(num_elem_stream,
                                                       sqKernel, 0u);

        cudaDeviceSynchronize();

        auto timer = DevTimer{};
        timer.tic();
        for (auto i = 0u; i < num_streams; ++i)
        {
            // Copy the input data to the device.
            auto const offset = i * num_elem_stream;
            checkError(cudaMemcpyAsync(input_dev,
                                       input_host.data() + offset,
                                       num_byte_stream,
                                       cudaMemcpyHostToDevice,
                                       streams[i].getStream()),
                       "copying data to device");
            // Execute the kernel.
            sqKernel<<<exec_params.grid_dim, exec_params.block_dim, 0, streams[i].getStream()>>>(
                input_dev, output_dev, num_elem_stream);
            // Copy the result from the device to the host.
            checkError(cudaMemcpyAsync(res_host.data() + offset,
                                       output_dev,
                                       num_byte_stream,
                                       cudaMemcpyDeviceToHost,
                                       streams[i].getStream()),
                       "copying data to host");
        }

        cudaDeviceSynchronize();
        auto const duration = timer.toc();

        // Clean up.
        cudaFree(input_dev);
        cudaFree(output_dev);

        if (!Detail::hasSameVal(std::span<float>{res_host.data(), res_host.size()},
                                init_val * init_val))
        {
            std::cerr << "Error: Kernel execution failed\n";
            std::exit(1);
        }

        return duration;
    }

    MilliSeconds zeroCopyExecute(unsigned num_elems)
    {
        auto constexpr init_val = 2.0f;
        auto const input_mapped = MappedVector<float>{std::vector<float>(num_elems, init_val)};
        auto output_mapped = MappedVector<float>{std::vector<float>(num_elems)};

        cudaDeviceSynchronize();
        auto timer = DevTimer{};
        timer.tic();

        auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);
        sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
            input_mapped.mappedData(), output_mapped.mappedData(), num_elems);

        cudaDeviceSynchronize();
        auto const duration = timer.toc();

        if (!Detail::hasSameVal(std::span<float>{output_mapped.data(), num_elems}, init_val * init_val))
        {
            std::cerr << "Error: Kernel execution failed\n";
            std::exit(1);
        }
        return duration;
    }

    PerfTestResult runPerfTest(unsigned num_rep)
    {
        auto constexpr num_elems = 1u << 24;
        auto perf_info = PerfTestResult{};

        for (auto i = 0u; i < num_rep; ++i)
        {
            perf_info["seq-pageable"].emplace_back(seqCopyExecutePageable(num_elems).count());
            perf_info["seq-unified"].emplace_back(seqCopyExecuteUnified(num_elems).count());
            perf_info["seq-pinned"].emplace_back(seqCopyExecutePinned(num_elems).count());
            perf_info["seq-zero-copy"].emplace_back(zeroCopyExecute(num_elems).count());
            perf_info["staged-concurrent"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 256u).count());
        }
        return perf_info;
    }

    PerfTestResult stagedCopyNumStreamsTest(unsigned num_rep)
    {
        auto constexpr num_elems = 1u << 24;
        auto perf_info = PerfTestResult{};

        for (auto i = 0u; i < num_rep; ++i)
        {
            perf_info["seq-pageable"].emplace_back(seqCopyExecutePageable(num_elems).count());
            perf_info["2"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 2u).count());
            perf_info["4"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 4u).count());
            perf_info["16"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 16u).count());
            perf_info["32"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 16u).count());
            perf_info["128"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 128u).count());
            perf_info["256"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 256u).count());
            perf_info["512"].emplace_back(
                stagedConcurrentCopyExecute(num_elems, 512u).count());
        }
        return perf_info;
    }

    namespace Detail
    {
        bool hasSameVal(std::span<float> vec, float val)
        {
            // Check if all the elements in the vector have the same value.
            return std::all_of(std::begin(vec), std::end(vec),
                               [val](float elem)
                               { return elem == val; });
        }
    } // Detail namespace.
} // Latency namespace.
