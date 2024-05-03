#include <iostream>
#include "pmpp/07_convolution/conv_1d.cuh"
#include "pmpp/02_vec_add/vec_add.cuh"
#include "pmpp/utils/dev_config.cuh"
#include "pmpp/utils/exec_config.cuh"
#include "pmpp/utils/host_timer.h"
#include "pmpp/08_stencil/stencil_1d.cuh"
#include "pmpp/03_color_gray_scale/color_gray_scale.cuh"
int main()
{
    try {
        //auto const color_file = std::string{"/home/shiras/Downloads/passphoto.jpg"};
        //auto const gray_file = std::string{"/home/shiras/Downloads/passphoto_grayx4000x6000.bin"};
        //PMPP::color2Gray(color_file, gray_file);
        //PMPP::CUDA::vecAddExample();
    }
    catch (std::exception const &e) {
        std::cout << "Exception is thrown." << std::endl;
        std::cout << e.what() << std::endl;
    }
    return 0;
}
