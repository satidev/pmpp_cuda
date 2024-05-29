#include "dev_config.cuh"
#include <iostream>

DeviceConfigSingleton::DeviceConfigSingleton()
{
    auto num_dev = 0;
    cudaGetDeviceCount(&num_dev);
    num_dev_ = static_cast<unsigned>(num_dev);
    setDevProperties();
}

void DeviceConfigSingleton::setDevProperties()
{
    dev_props_.reserve(num_dev_);

    for (auto dev_id = 0; dev_id < num_dev_; dev_id++) {

        auto dev_prop = cudaDeviceProp{};
        cudaGetDeviceProperties(&dev_prop, dev_id);
        auto dev_prop_struct = DeviceProperties{};

        // Device info.
        dev_prop_struct.device_id = dev_id;
        dev_prop_struct.device_name = dev_prop.name;

        // Thread resources.
        dev_prop_struct.num_sm = dev_prop.multiProcessorCount;
        dev_prop_struct.global_mem_size = dev_prop.totalGlobalMem;
        dev_prop_struct.constant_mem_size = dev_prop.totalConstMem;
        dev_prop_struct.max_threads_per_sm = dev_prop.maxThreadsPerMultiProcessor;
        dev_prop_struct.max_cuncurrent_threads_per_device = dev_prop.maxThreadsPerMultiProcessor *
            dev_prop.multiProcessorCount;
        dev_prop_struct.max_threads_per_block = dev_prop.maxThreadsPerBlock;
        dev_prop_struct.max_shared_mem_per_sm = dev_prop.sharedMemPerMultiprocessor;
        dev_prop_struct.max_shared_mem_per_block = dev_prop.sharedMemPerBlock;
        dev_prop_struct.max_regs_per_block = dev_prop.regsPerBlock;
        dev_prop_struct.max_regs_per_sm = dev_prop.regsPerMultiprocessor;
        dev_prop_struct.max_threads_per_warp = dev_prop.warpSize;
        dev_prop_struct.max_blocks_per_sm = dev_prop.maxBlocksPerMultiProcessor;
        dev_prop_struct.warp_size = dev_prop.warpSize;
        dev_prop_struct.memory_clock_rate = dev_prop.memoryClockRate;
        dev_prop_struct.memory_bus_width = dev_prop.memoryBusWidth;
        dev_prop_struct.peak_memory_bandwidth =
            2.0f * static_cast<float>(dev_prop.memoryClockRate) *
                (static_cast<float>(dev_prop.memoryBusWidth) / 8.0f) * 1e-6f;
        dev_prop_struct.can_map_host_memory = static_cast<bool>(dev_prop.canMapHostMemory);

        dev_props_.push_back(dev_prop_struct);
    }
}

void DeviceConfigSingleton::printDeviceProperties(unsigned dev_id) const
{
    if (dev_id >= num_dev_) {
        throw std::invalid_argument{"Invalid device ID."};
    }
    auto const dev_prop = dev_props_[dev_id];
    std::cout << "Device ID: " << dev_prop.device_id << std::endl;
    std::cout << "Device name: " << dev_prop.device_name << std::endl;

    std::cout << "Number of SMs (streaming multi-processor): " << dev_prop.num_sm << std::endl;
    std::cout << "Maximum concurrent threads per device: "
              << dev_prop.max_cuncurrent_threads_per_device
              << std::endl;
    std::cout << "Maximum active threads per SM: " << dev_prop.max_threads_per_sm << std::endl;
    std::cout << "Maximum active threads per block: " << dev_prop.max_threads_per_block <<
              std::endl;
    std::cout << "Maximum threads per warp: " << dev_prop.max_threads_per_warp << std::endl;
    std::cout << "Warp size: " << dev_prop.warp_size << std::endl;
    std::cout << "Maximum blocks per SM: " << dev_prop.max_blocks_per_sm << std::endl;

    std::cout << "Global memory size: " << dev_prop.global_mem_size << " bytes" << std::endl;
    std::cout << "Constant memory size: " << dev_prop.constant_mem_size << " bytes" << std::endl;

    std::cout << "Maximum shared memory per SM: " << dev_prop.max_shared_mem_per_sm <<
              " bytes" << std::endl;
    std::cout << "Maximum shared memory per block: " << dev_prop.max_shared_mem_per_block <<
              " bytes" << std::endl;
    std::cout << "Maximum registers per block: " << dev_prop.max_regs_per_block <<
              " bytes " << std::endl;
    std::cout << "Maximum registers per SM: " << dev_prop.max_regs_per_sm <<
              " bytes" << std::endl;

    std::cout << "Memory clock rate: " << dev_prop.memory_clock_rate <<
              " KHz" << std::endl;
    std::cout << "Memory bus width: " << dev_prop.memory_bus_width <<
              " bits" << std::endl;
    std::cout << "Peak memory bandwidth: " << dev_prop.peak_memory_bandwidth <<
              " GB/sec" << std::endl;
    std::cout << "Can map host memory?: ";
    if(dev_prop.can_map_host_memory) {
        std::cout << "Yes" << std::endl;
    }
    else {
        std::cout << "No" << std::endl;
    }
}

