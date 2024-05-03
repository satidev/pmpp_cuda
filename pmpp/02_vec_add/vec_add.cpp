#include "vec_add.h"
#include <stdexcept>
#include <algorithm>
#include <iterator>

namespace PMPP
{
std::vector<float> vecAdd(std::vector<float> const &first,
                          std::vector<float> const &sec)
{
    if (std::size(first) != std::size(sec)) {
        throw std::invalid_argument{"Size should be equal"};
    }
    auto res = std::vector<float>{};
    res.reserve(first.size());

    std::transform(std::begin(first), std::end(first), std::begin(sec),
                   std::back_inserter(res), std::plus<float>{});
    return res;
}

}// Numeric namespace.
