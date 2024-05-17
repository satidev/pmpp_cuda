#include "color_gray_scale.cuh"
#include <iostream>
#include <vector>
#include <stdexcept>
#include "../utils/dev_vector.cuh"
#include "../utils/dev_vector_factory.cuh"
#include <dlib/image_io.h>

namespace PMPP
{
__global__ void color2GrayKernel(unsigned char const *color_data,
                                 float *gray_data,
                                 unsigned num_rows,
                                 unsigned num_cols)
{
    auto const row{blockIdx.y * blockDim.y + threadIdx.y};
    auto const col{blockIdx.x * blockDim.x + threadIdx.x};

    if (row < num_rows && col < num_cols) {
        auto const idx{row * num_cols + col};

        auto const r{static_cast<float>(color_data[3 * idx])};
        auto const g{static_cast<float>(color_data[3 * idx + 1])};
        auto const b{static_cast<float>(color_data[3 * idx + 2])};

        gray_data[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void color2Gray(std::string const &color_file,
                std::string const &gray_file)
{
    std::cout << "Color to gray conversion" << std::endl;
    auto const img_data = loadRGBImage(color_file);
    auto const color_dev = DevVectorFactory::create(img_data.pixel_data);
    auto gray_dev = DevVector<float>{img_data.num_rows * img_data.num_cols};

    auto const block_size = dim3{32u, 32u};
    auto const grid_size = dim3{(img_data.num_cols + block_size.x - 1) / block_size.x,
                                (img_data.num_rows + block_size.y - 1) / block_size.y};
    color2GrayKernel<<<grid_size, block_size>>>(color_dev.data(),
                                                gray_dev.data(),
                                                img_data.num_rows,
                                                img_data.num_cols);
    checkErrorKernel("color2GrayKernel", true);

    auto const gray_host = HostDevCopy::hostCopy(gray_dev);

    // Copy to a binary file.
    auto const gray_file_ptr = std::fopen(gray_file.c_str(), "wb");
    if (gray_file_ptr == nullptr) {
        throw std::runtime_error{"Cannot open file for writing\n"};
    }
    // Write header information.
    std::fwrite(gray_host.data(), sizeof(float),
                img_data.num_rows * img_data.num_cols, gray_file_ptr);
    // Write pixel data.
    std::fwrite(gray_host.data(), sizeof(float),
                img_data.num_rows * img_data.num_cols, gray_file_ptr);
    std::fclose(gray_file_ptr);
}

RGBImage loadRGBImage(std::string const &img_filename)
{
    auto img = dlib::array2d<dlib::rgb_pixel>{};
    dlib::load_image(img, img_filename);
    if (img.size() == 0) {
        throw std::invalid_argument{"Cannot load image\n"};
    }

    auto const num_rows = static_cast<unsigned>(img.nr());
    auto const num_cols = static_cast<unsigned>(img.nc());

    auto img_data = std::vector<unsigned char>{};
    img_data.reserve(num_rows * num_cols * 3u);
    for (auto const &pixel: img) {
        img_data.push_back(pixel.red);
        img_data.push_back(pixel.green);
        img_data.push_back(pixel.blue);
    }

    return RGBImage{num_rows, num_cols, std::move(img_data)};
}

}// PMPP namespace.