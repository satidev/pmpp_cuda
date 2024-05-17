#ifndef COPY_EXECUTE_LATENCY_CUH
#define COPY_EXECUTE_LATENCY_CUH

// Computes the latency of the memory transfer between the
// host and the device memory and kernel execution.
namespace CopyExecuteLatency
{
    // Sequential data copy and kernel execution (pageable memory).
    float seqCopyExecutePageable(unsigned num_elems);

    // Sequential data copy and kernel execution (unified memory).
    float seqCopyExecuteUnified(unsigned num_elems);

    // Sequential data copy and kernel execution (pinned memory).
    float seqCopyExecutePinned(unsigned num_elems);

    // Staged concurrent copy and execution.
    // This is only possible by using pinned memory and streams.
    float stagedConcurrentCopyExecute(unsigned num_elems, unsigned num_streams);

    float zeroCopyExecute(unsigned num_elems);

    void runPerfTest();
}// CopyExecuteLatency namespace.

#endif //COPY_EXECUTE_LATENCY_CUH


