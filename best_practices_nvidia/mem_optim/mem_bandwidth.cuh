#ifndef MEM_BANDWIDTH_CUH
#define MEM_BANDWIDTH_CUH

// Computes the bandwidth of the memory transfer between the host and the device.
namespace MemoryBandwidth
{
    // Transfer data from STL vector to device memory.
    float pageableMem(unsigned num_elems, unsigned num_reps);
    // Transfer data from the pinned host memory to device memory.
    float pinnedMem(unsigned num_elems, unsigned num_reps);
    // First register the memory and then transfer data from the pinned host memory to device memory.
    float pinnedMemRegister(unsigned num_elems, unsigned num_reps);

    void runPerfTest();
}// MemoryBandwidth namespace.

#endif //MEM_BANDWIDTH_CUH
