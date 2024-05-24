#include <iostream>
#include "pmpp/07_convolution/conv_1d.cuh"
#include "utils/dev_config.cuh"
#include "utils/exec_config.cuh"
#include "utils/host_timer.h"
#include "pmpp/08_stencil/stencil_1d.cuh"
#include "pmpp/03_color_gray_scale/color_gray_scale.cuh"
#include "pmpp/03_05_06_mat_mul/mat_mul.cuh"
#include "best_practices_nvidia/mem_optim/mem_bandwidth.cuh"
#include "best_practices_nvidia/mem_optim/copy_execute_latency.cuh"
#include "best_practices_nvidia/mat_transpose/transpose.cuh"
#include "pmpp/02_vec_add/vec_add.cuh"

int main()
{
    try {
        auto const color_file = std::string{"/home/shiras/Downloads/passphoto.jpg"};
        auto const gray_file = std::string{"/home/shiras/Downloads/passphoto_grayx4000x6000.bin"};
        //PMPP::color2Gray(color_file, gray_file);
        //PMPP::vecAddPerfTest();
        //PMPP::matMulPerfTest();
        //Bandwidth::bandwidthTest();
        //CopyExecuteLatency::runPerfTest();
        BPNV::transposePerfTest();
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
