#include "color_gray_scale.cuh"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include "../utils/dev_vector.cuh"
#include <array>

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

    // Read color image file using openCV.
    auto const cv_img = cv::imread(color_file, cv::IMREAD_COLOR);
    if (cv_img.empty()) {
        throw std::runtime_error{"Cannot read image file\n"};
    }

    // Copy pixel values to a STL vector.

    auto const num_rows = static_cast<unsigned>(cv_img.rows);
    auto const num_cols = static_cast<unsigned>(cv_img.cols);
    auto const num_pixels = num_rows * num_cols;
    auto img_data = std::vector<unsigned char>(num_pixels * 3u);
    std::copy(cv_img.data, cv_img.data + cv_img.rows * cv_img.cols * 3u, std::begin(img_data));

    auto const color_dev = DevVector<unsigned char>{img_data};
    auto gray_dev = DevVector<float>{num_pixels};

    auto const block_size = dim3{32u, 32u};
    auto const grid_size = dim3{(num_cols + block_size.x - 1) / block_size.x,
                                (num_rows + block_size.y - 1) / block_size.y};
    color2GrayKernel<<<grid_size, block_size>>>(color_dev.data(),
                                                gray_dev.data(),
                                                num_rows,
                                                num_cols);
    checkErrorKernel("color2GrayKernel", true);

    auto const gray_host = gray_dev.hostCopy();

    // Copy to a binary file.
    auto const gray_file_ptr = std::fopen(gray_file.c_str(), "wb");
    if (gray_file_ptr == nullptr) {
        throw std::runtime_error{"Cannot open file for writing\n"};
    }
    // Write header information.
    std::fwrite(gray_host.data(), sizeof(float), num_pixels, gray_file_ptr);
    std::fwrite(gray_host.data(), sizeof(float), num_pixels, gray_file_ptr);
    std::fclose(gray_file_ptr);
}

}// PMPP namespace.