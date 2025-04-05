#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <vector>

namespace PMPP::CUDA
{
    std::vector<unsigned> histogram(std::vector<bool> const &data_host);
    std::vector<unsigned> histogramPrivatization(std::vector<bool> const &data_host);
    std::vector<unsigned> histogramPrivateShared(std::vector<bool> const &data_host);

    enum class CoarseningStrategy
    {
        CONTIGUOUS_PARTITIONING,
        INTERLEAVED_PARTITIONING
    };
    std::vector<unsigned> histogramPrivateSharedCoarse(std::vector<bool> const &data_host,
                                                       CoarseningStrategy strategy);
    // Interleaved coarsening + aggregation.
    std::vector<unsigned> histogramPrivateSharedCoarseAggr(std::vector<bool> const &data_host);
} // Numeric::CUDA namespace.

#endif // HISTOGRAM_CUH
