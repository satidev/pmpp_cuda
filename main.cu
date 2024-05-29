#include <iostream>
#include "bpnv/mat_transpose/perf_test.cuh"
#include <dlib/cmd_line_parser.h>
#include "perf_test_analyzer.h"
#include "bpnv/mem_optim/mem_bandwidth.cuh"
#include "bpnv/mem_optim/copy_execute_latency.cuh"
#include "utils/dev_config.cuh"

int main(int argc, char** argv)
{
    try {
        if(argc < 2) {
            std::cerr << "Usage: ./run_perf_tests -f <operation_name> -n <num_repetitions> -o <output_dir>" << std::endl;
            return 1;
        }

        auto parser = dlib::command_line_parser{};
        parser.add_option("h", "Display this message.");
        parser.add_option("a", "Action/operation whose performance needs to analyzed.", 1);
        parser.add_option("n", "Number of repetitions.", 1);
        parser.add_option("o", "Output directory name to save plots.", 1);
        parser.parse(argc, argv);

        if (parser.option("h")) {
            std::cout << "Usage: ./run_perf_tests -f <operation_name> -n <num_iterations> -o <output_dir>" << std::endl;
            return 0;
        }

        auto const action = parser.option("a").argument();

        auto num_repetitions = 10u;
        if (parser.option("n")) {
            num_repetitions = std::stoi(parser.option("n").argument());
        }

        auto output_dir = std::string{};
        if (parser.option("o")) {
            output_dir = parser.option("o").argument();
        }
        if(action == "know-your-gpu") {
            auto const& dev_config = DeviceConfigSingleton::getInstance();
            std::cout << "Number of CUDA devices: " << dev_config.numDevices() << std::endl;
            std::cout << "Properties for device 0:" << std::endl;
            dev_config.printDeviceProperties(0);
        }
        else if (action == "mat-transpose") {
            auto const perf_info = BPNV::transposePerfTest(num_repetitions);
            auto const analyzer = PerfTestAnalyzer{perf_info};
            analyzer.plotPerfMetric(output_dir, "Kernel execution time", "Milli seconds");
            analyzer.plotPerfBoostInfo(output_dir, "naive");
        }
        else if (action == "mem-bandwidth") {
            using namespace BPNV::MemoryBandwidth;
            auto const perf_info = runPerfTest(num_repetitions);
            auto const analyzer = PerfTestAnalyzer{perf_info};
            analyzer.plotPerfMetric(output_dir, "Memory bandwidth", "GB/sec");

        }
        else if(action == "copy-kernel-copy") {
            using namespace BPNV::CopyExecuteLatency;
            auto const perf_info = runPerfTest(num_repetitions);
            auto const analyzer = PerfTestAnalyzer{perf_info};
            analyzer.plotPerfMetric(output_dir, "Copy-Kernel-Copy", "milli seconds");
            analyzer.plotPerfBoostInfo(output_dir, "seq-pageable");
        }
        else if(action == "staged-copy-streams") {
            using namespace BPNV::CopyExecuteLatency;
            auto const perf_info = stagedCopyNumStreamsTest(num_repetitions);
            auto const analyzer = PerfTestAnalyzer{perf_info};
            analyzer.plotPerfMetric(output_dir, "Staged copy: number of streams", "milli seconds");
            analyzer.plotPerfBoostInfo(output_dir, "seq-pageable");
        }
        else {
            std::cerr << "Invalid action operation." << std::endl;
            std::cerr << "Usage: ./run_perf_tests -a <operation_name> -n <num_iterations> -o <output_dir>" << std::endl;
        }
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
