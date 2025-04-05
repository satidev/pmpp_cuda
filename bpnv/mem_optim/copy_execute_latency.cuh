#ifndef COPY_EXECUTE_LATENCY_CUH
#define COPY_EXECUTE_LATENCY_CUH

#include "../../utils/perf.cuh"
#include <span>

// Computes the latency of the memory transfer between the
// host and the device memory and kernel execution.
namespace BPNV::CopyExecuteLatency
{
    // Sequential data copy and kernel execution (pageable memory).
    MilliSeconds seqCopyExecutePageable(unsigned num_elems);

    // Sequential data copy and kernel execution (unified memory).
    MilliSeconds seqCopyExecuteUnified(unsigned num_elems);

    // Sequential data copy and kernel execution (pinned memory).
    MilliSeconds seqCopyExecutePinned(unsigned num_elems);

    // Staged concurrent copy and execution.
    // This is only possible by using pinned memory and streams.
    MilliSeconds stagedConcurrentCopyExecute(unsigned num_elems, unsigned num_streams);

    MilliSeconds zeroCopyExecute(unsigned num_elems);

    PerfTestResult runPerfTest(unsigned num_rep);

    PerfTestResult stagedCopyNumStreamsTest(unsigned num_rep);

    namespace Detail
    {
        bool hasSameVal(std::span<float> vec, float val);
    } // Detail namespace.
} // CopyExecuteLatency namespace.

#endif // COPY_EXECUTE_LATENCY_CUH
