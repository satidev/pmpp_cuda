#include <numeric>
#include "copy_execute_latency.cuh"
#include "../../pmpp/utils/dev_timer.cuh"
#include "../../pmpp/utils/check_error.cuh"
#include "../../pmpp/utils/exec_config.cuh"

namespace CopyExecuteLatency
{
__global__ void sqKernel(float const *ip, float *op,
                         unsigned num_elems)
{
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        op[idx] = ip[idx] * ip[idx];
    }
}

float seqCopyExecutePageable(unsigned num_elems)
{
    // Allocate input data in host memory.
    auto constexpr init_val = 2.0f;
    auto const input_data_host = std::vector<float>(num_elems, init_val);

    // Allocate input data in device memory and transfer the data.
    auto const num_bytes = num_elems * sizeof(float);
    auto input_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&input_dat_dev), num_bytes),
               "allocating device memory for input data");

    // Allocate device memory for the result.
    auto output_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&output_dat_dev), num_bytes),
               "allocating device memory for output data");

    // Allocate the host memory for the result.
    auto res_host = std::vector<float>(num_elems);
    auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);

    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();

    // Copy the input data to the device.
    checkError(cudaMemcpy(input_dat_dev,
                          input_data_host.data(),
                          num_bytes,
                          cudaMemcpyHostToDevice),
               "copying data to device");
    // Execute the kernel.
    sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        input_dat_dev, output_dat_dev, num_elems);

    // Copy the result from the device to the host.
    checkError(cudaMemcpy(res_host.data(),
                          output_dat_dev,
                          num_bytes,
                          cudaMemcpyDeviceToHost),
               "copying data to host");

    cudaDeviceSynchronize();
    auto const duration = timer.toc();
    cudaFree(input_dat_dev);
    cudaFree(output_dat_dev);

    // Check the output results by computing the mean.
    auto const mean = std::accumulate(std::begin(res_host), std::end(res_host), 0.0f) /
        static_cast<float>(num_elems);
    if (mean != init_val * init_val) {
        std::cerr << "Error: Kernel execution failed\n";
        std::exit(1);
    }

    return duration;
}

float seqCopyExecuteUnified(unsigned num_elems)
{

    auto constexpr init_val = 2.0f;

    // Allocate input data in unified memory that can be
    // accessed by both the host and the device.
    auto const num_bytes = num_elems * sizeof(float);
    auto input_data = static_cast<float *>(nullptr);
    checkError(cudaMallocManaged(reinterpret_cast<void **>(&input_data), num_bytes),
               "allocating unified memory for input data");
    for (auto i = 0u; i < num_elems; ++i) {
        input_data[i] = init_val;
    }

    //checkError(cudaMemset(input_data, init_val, num_bytes),
    //           "initializing unified memory for input data");

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
    // Check the output results by computing the mean.
    auto mean = 0.0f;
    for (auto i = 0u; i < num_elems; ++i) {
        mean += output_data[i];
    }
    mean /= static_cast<float>(num_elems);
    if (mean != init_val * init_val) {
        std::cout << "Mean: " << mean << std::endl;
        std::cerr << "Error: Kernel execution failed\n";
        std::exit(1);
    }

    // Clean up.
    cudaFree(input_data);
    cudaFree(output_data);

    return duration;
}

float seqCopyExecutePinned(unsigned num_elems)
{
    // Allocate input data in host memory.
    auto constexpr init_val = 2.0f;
    auto const input_data_host = std::vector<float>(num_elems, init_val);
    checkError(cudaHostRegister((void *) input_data_host.data(), num_elems * sizeof(float),
                                cudaHostRegisterDefault),
               "registering input data host memory");

    // Allocate input data in device memory and transfer the data.
    auto const num_bytes = num_elems * sizeof(float);
    auto input_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&input_dat_dev), num_bytes),
               "allocating device memory for input data");

    // Allocate device memory for the result.
    auto output_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&output_dat_dev), num_bytes),
               "allocating device memory for output data");

    // Allocate the host memory for the result.
    auto res_host = std::vector<float>(num_elems);
    checkError(cudaHostRegister((void *) res_host.data(), num_elems * sizeof(float),
                                cudaHostRegisterDefault),
               "registering result host memory");
    auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);

    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();

    // Copy the input data to the device.
    checkError(cudaMemcpyAsync(input_dat_dev,
                               input_data_host.data(),
                               num_bytes,
                               cudaMemcpyHostToDevice),
               "copying data to device");
    // Execute the kernel.
    sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        input_dat_dev, output_dat_dev, num_elems);

    // Copy the result from the device to the host.
    checkError(cudaMemcpyAsync(res_host.data(),
                               output_dat_dev,
                               num_bytes,
                               cudaMemcpyDeviceToHost),
               "copying data to host");

    cudaDeviceSynchronize();
    auto const duration = timer.toc();

    // Clean up.
    cudaFree(input_dat_dev);
    cudaFree(output_dat_dev);
    checkError(cudaHostUnregister((void *) input_data_host.data()),
               "unregistering input data host memory");
    checkError(cudaHostUnregister((void *) res_host.data()),
               "unregistering input data host memory");

    // Check the output results by computing the mean.
    auto const mean = std::accumulate(std::begin(res_host), std::end(res_host), 0.0f) /
        static_cast<float>(num_elems);
    if (mean != init_val * init_val) {
        std::cerr << "Error: Kernel execution failed\n";
        std::exit(1);
    }

    return duration;
}

float stagedConcurrentCopyExecute(unsigned num_elems, unsigned num_streams)
{
    // Allocate input data in host memory.
    auto constexpr init_val = 2.0f;
    auto const input_data_host = std::vector<float>(num_elems, init_val);
    checkError(cudaHostRegister((void *) input_data_host.data(), num_elems * sizeof(float),
                                cudaHostRegisterDefault),
               "registering input data host memory");

    // Allocate input data in device memory and transfer the data.
    auto const num_elem_stream = num_elems / num_streams;
    auto const num_byte_stream = num_elem_stream * sizeof(float);
    auto input_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&input_dat_dev), num_byte_stream),
               "allocating device memory for input data");

    // Allocate device memory for the result.
    auto output_dat_dev = static_cast<float *>(nullptr);
    checkError(cudaMalloc(reinterpret_cast<void **>(&output_dat_dev), num_byte_stream),
               "allocating device memory for output data");

    // Allocate the host memory for the result.
    auto res_host = std::vector<float>(num_elems);
    checkError(cudaHostRegister((void *) res_host.data(), num_elems * sizeof(float),
                                cudaHostRegisterDefault),
               "registering result host memory");

    // Allocate streams.
    auto streams = std::vector<cudaStream_t>(num_streams);
    for (auto &stream: streams) {
        checkError(cudaStreamCreate(&stream), "creating stream");
    }

    auto const exec_params = ExecConfig::getParams(num_elem_stream,
                                                   sqKernel, 0u);

    cudaDeviceSynchronize();

    auto timer = DevTimer{};
    timer.tic();
    for (auto i = 0u; i < num_streams; ++i) {
        // Copy the input data to the device.
        auto const offset = i * num_elem_stream;
        checkError(cudaMemcpyAsync(input_dat_dev,
                                   input_data_host.data() + offset,
                                   num_byte_stream,
                                   cudaMemcpyHostToDevice,
                                   streams[i]),
                   "copying data to device");
        // Execute the kernel.
        sqKernel<<<exec_params.grid_dim, exec_params.block_dim, 0, streams[i]>>>(
            input_dat_dev, output_dat_dev, num_elem_stream);
        // Copy the result from the device to the host.
        checkError(cudaMemcpyAsync(res_host.data() + offset,
                                   output_dat_dev,
                                   num_byte_stream,
                                   cudaMemcpyDeviceToHost,
                                   streams[i]),
                   "copying data to host");
    }

    cudaDeviceSynchronize();
    auto const duration = timer.toc();

    // Clean up.
    for (auto &stream: streams) {
        cudaStreamDestroy(stream);
    }
    cudaFree(input_dat_dev);
    cudaFree(output_dat_dev);
    checkError(cudaHostUnregister((void *) input_data_host.data()),
               "unregistering input data host memory");
    checkError(cudaHostUnregister((void *) res_host.data()),
               "unregistering input data host memory");

    // Check the output results by computing the mean.
    auto const mean = std::accumulate(std::begin(res_host), std::end(res_host), 0.0f) /
        static_cast<float>(num_elems);
    if (mean != init_val * init_val) {
        std::cerr << "Error: Kernel execution failed\n";
        std::exit(1);
    }

    return duration;
}

float zeroCopyExecute(unsigned num_elems)
{
    // Allocate input data in host memory.
    auto constexpr init_val = 2.0f;
    auto const input_data_host = std::vector<float>(num_elems, init_val);

    // Check the device properties.
    auto dev_prop = cudaDeviceProp{};
    checkError(cudaGetDeviceProperties(&dev_prop, 0), "getting device properties");
    if (!dev_prop.canMapHostMemory) {
        std::cerr << "Error: Zero copy memory is not supported\n";
        std::exit(1);
    }

    checkError(cudaSetDeviceFlags(cudaDeviceMapHost),
               "setting device flags for zero copy memory");
    checkError(cudaHostRegister((void *) input_data_host.data(), num_elems * sizeof(float),
                                cudaHostRegisterMapped),
               "registering input data host memory");
    auto input_data_mapped = static_cast<float *>(nullptr);
    checkError(cudaHostGetDevicePointer((void **) &input_data_mapped,
                                        (void *) input_data_host.data(), 0),
               "getting device pointer for mapped memory");

    // Allocate the host memory for the result.
    auto res_dev = static_cast<float *>(nullptr);
    checkError(cudaMallocManaged((void **) &res_dev, num_elems * sizeof(float)),
               "allocating device memory for output data");

    cudaDeviceSynchronize();
    auto timer = DevTimer{};
    timer.tic();

    auto const exec_params = ExecConfig::getParams(num_elems, sqKernel, 0u);
    sqKernel<<<exec_params.grid_dim, exec_params.block_dim>>>(
        input_data_mapped, res_dev, num_elems);

    cudaDeviceSynchronize();
    auto const duration = timer.toc();


    // Check the output results by computing the mean.
    auto mean = 0.0f;
    for (auto i = 0u; i < num_elems; ++i) {
        mean += res_dev[i];
    }
    mean /= static_cast<float>(num_elems);
    if (mean != init_val * init_val) {
        std::cout << "Mean: " << mean << std::endl;
        std::cerr << "Error: Kernel execution failed\n";
        std::exit(1);
    }



    // Clean up.
    // Deregister the host memory.
    checkError(cudaHostUnregister((void *) input_data_host.data()),
               "unregistering input data host memory");
    cudaFree(res_dev);


    return duration;
}

void runPerfTest()
{
    auto constexpr num_elems = 1u << 24;
    std::cout << "Sequential copy and execute (pageable):: "
              << seqCopyExecutePageable(num_elems) << " sec" << std::endl;
    std::cout << "Sequential copy and execute (pinned):: "
              << seqCopyExecutePinned(num_elems) << " sec" << std::endl;
    std::cout << "Sequential copy and execute (unified):: "
              << seqCopyExecuteUnified(num_elems) << " sec" << std::endl;

    std::cout << "Zero-copy and execute:: "
              << zeroCopyExecute(num_elems) << " sec" << std::endl;

    auto constexpr num_streams = 256u;
    std::cout << "Staged concurrent copy and execute :: "
              << stagedConcurrentCopyExecute(num_elems, num_streams) << " sec" << std::endl;
}

}// Latency namespace.
