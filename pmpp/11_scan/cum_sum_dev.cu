#include "cum_sum_dev.cuh"
#include "../utils/dev_vector.cuh"
#include "../utils/dev_vector_factory.cuh"
#include <stdexcept>

namespace PMPP
{
    __global__ void koggeStoneKernel(float const *input,
                                     float *output)
    {
        // Only a single block is used.
        // The number of elements in the input array is equal to the block size.

        // Copy input data to a shared memory.
        extern __shared__ float xy_sm[];
        xy_sm[threadIdx.x] = input[threadIdx.x];
        __syncthreads();

        // Reduction operation for each output element.
        // After stride iteration, xy_sm[threadIdx.x] contains the sum of up to
        // 2^(stride) elements at and before threadIdx.x-th location.
        for (auto stride = 1u; stride < blockDim.x; stride *= 2)
        {

            auto temp{0.0f};
            if (threadIdx.x >= stride)
            {
                temp = xy_sm[threadIdx.x - stride] + xy_sm[threadIdx.x];
            }

            // To avoid writer-after-reader data hazard.
            // write to a register before a prior instruction reads it.
            // Barrier synchronization is needed to ensure that all threads have
            // written to the shared memory before any thread reads from it.
            __syncthreads();

            if (threadIdx.x >= stride)
            { // The accumulation is finished.
                xy_sm[threadIdx.x] = temp;
            }

            __syncthreads();
        }
        // Copy the result to the output array.
        output[threadIdx.x] = xy_sm[threadIdx.x];
    }

    std::vector<float> cumSumDev(std::vector<float> const &vec,
                                 ScanAlgorithm algo)
    {
        if (std::empty(vec))
            return std::vector<float>{};

        // Input size should be a multiple of 32.
        if (std::size(vec) % 32 != 0)
            throw std::invalid_argument{"Input size should be a multiple of 32."};

        auto const vec_dev = DevVectorFactory::create(vec);
        auto const num_elems = static_cast<unsigned>(std::size(vec));
        auto res_dev = DevVector<float>{num_elems};

        auto const block_size = num_elems;
        auto const grid_size = 1u;
        auto const shared_mem_size = block_size * sizeof(float);
        switch (algo)
        {
        case ScanAlgorithm::KOGGE_STONE:
            koggeStoneKernel<<<grid_size, block_size, shared_mem_size>>>(
                vec_dev.data(), res_dev.data());
            break;
        }

        return HostDevCopy::hostCopy(res_dev);
    }
} // PMPP namespace.
