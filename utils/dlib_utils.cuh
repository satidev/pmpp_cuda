#ifndef DLIB_UTILS_CUH
#define DLIB_UTILS_CUH

#include "dlib/matrix.h"

namespace DlibUtils
{
    template <typename T>
    dlib::matrix<T> constMat(int num_rows, int num_cols, T val)
    {
        auto mat = dlib::matrix<T>(num_rows, num_cols);
        mat = val;
        return mat;
    }
} // DlibUtils namespace.

#endif // DLIB_UTILS_CUH
