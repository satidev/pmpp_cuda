#ifndef DEV_CONFIG_CUH
#define DEV_CONFIG_CUH

#include <vector>
#include <string>

struct DeviceProperties
{
    // Device info.
    size_t device_id;
    std::string device_name;
    std::string compute_capability;

    // Thread resources.
    size_t num_sm;
    size_t max_threads_per_sm;
    size_t max_cuncurrent_threads_per_device;
    size_t max_threads_per_warp;
    size_t max_threads_per_block;
    size_t warp_size;
    size_t max_blocks_per_sm;

    // Memory resources.
    size_t global_mem_size;
    size_t constant_mem_size;
    size_t max_shared_mem_per_sm;
    size_t max_shared_mem_per_block;
    size_t max_regs_per_block;
    size_t max_regs_per_sm;
    size_t memory_clock_rate;
    size_t memory_bus_width;
    float peak_memory_bandwidth;
    bool can_map_host_memory;
    size_t l2_cache_size;
};

class DeviceConfigSingleton
{
private:
    unsigned num_dev_{0u};
    std::vector<DeviceProperties> dev_props_;

public:
    DeviceConfigSingleton(DeviceConfigSingleton const &) = delete;
    DeviceConfigSingleton &operator=(DeviceConfigSingleton const &) = delete;
    DeviceConfigSingleton(DeviceConfigSingleton &&) = delete;
    DeviceConfigSingleton &operator=(DeviceConfigSingleton &&) = delete;
    ~DeviceConfigSingleton() = default;

    static DeviceConfigSingleton &getInstance()
    {
        auto static instance = DeviceConfigSingleton{};
        return instance;
    }

    [[nodiscard]] unsigned numDevices() const
    {
        return num_dev_;
    }

    [[nodiscard]] DeviceProperties const &getDevProps(unsigned dev_id = 0u) const
    {
        return dev_props_[dev_id];
    }

    void printDeviceProperties(unsigned dev_id = 0u) const;

private:
    DeviceConfigSingleton();
    void setDevProperties();
};


#endif //DEV_CONFIG_CUH
