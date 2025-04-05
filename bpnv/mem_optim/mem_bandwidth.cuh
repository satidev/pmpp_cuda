#ifndef MEM_BANDWIDTH_CUH
#define MEM_BANDWIDTH_CUH

#include "../../utils/perf.cuh"

// Computes the bandwidth of the memory transfer between the host and the device.
namespace BPNV::MemoryBandwidth
{
    // Transfer data from STL vector to device memory.
    float pageableMem(unsigned num_elems, unsigned num_transfers);

    // Transfer data from the pinned memory to device memory.
    float pinnedMem(unsigned num_elems, unsigned num_transfers);

    PerfTestResult runPerfTest(unsigned num_rep);
} // MemoryBandwidth namespace.

#endif // MEM_BANDWIDTH_CUH
