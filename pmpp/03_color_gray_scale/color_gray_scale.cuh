#ifndef COLOR_GRAY_SCALE_CUH
#define COLOR_GRAY_SCALE_CUH

#include <string>
#include <vector>

namespace PMPP
{
    struct RGBImage
    {
        unsigned num_rows;
        unsigned num_cols;
        std::vector<unsigned char> pixel_data;
    };

    RGBImage loadRGBImage(std::string const &img_filename);

    void color2Gray(std::string const &color_file,
                    std::string const &gray_file);
}

#endif //COLOR_GRAY_SCALE_CUH


