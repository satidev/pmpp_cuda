#ifndef EIGEN_UTILS_H
#define EIGEN_UTILS_H

#include <vector>
#include "Eigen/Dense"
#include <iostream>
#include <stdexcept>

namespace EigenUtils
{
template<typename T>
std::vector<T> toVec(Eigen::MatrixX<T> const &mat);

template<typename T>
Eigen::MatrixX<T> toMat(std::vector<T> const &vec,
                        unsigned num_rows, unsigned num_cols);

template<typename T>
std::vector<T> toVec(Eigen::MatrixX<T> const &mat)
{
    auto const num_elems = mat.rows() * mat.cols();
    auto vec = std::vector<T>{};
    vec.reserve(num_elems);
    for (auto row = 0; row < mat.rows(); row++)
        for (auto col = 0; col < mat.cols(); col++)
            vec.push_back(mat(row, col));
    return vec;
}

template<typename T>
Eigen::MatrixX<T> toMat(std::vector<T> const &vec,
                        unsigned num_rows, unsigned num_cols)
{
    if (vec.size() != num_rows * num_cols) {
        throw std::invalid_argument{"The number of elements should be consistent."};
    }
    auto mat = Eigen::MatrixX<T>(num_rows, num_cols);
    for (auto row = 0; row < mat.rows(); row++)
        for (auto col = 0; col < mat.cols(); col++)
            mat(row, col) = vec[row * num_cols + col];

    return mat;
}

}// EigenUtils namespace.

#endif //EIGEN_UTILS_H
