#include "vec_add.h"
#include <stdexcept>
#include <algorithm>

namespace Numeric
{
std::vector<float> vecAdd(std::vector<float> const &first,
                          std::vector<float> const &sec)
{
    if (first.size() != sec.size()) {
        throw std::invalid_argument{"Size should be equal"};
    }
    auto res = std::vector<float>{};
    res.reserve(first.size());

    std::transform(std::begin(first), std::end(first), std::begin(sec),
                   std::back_inserter(res), [](float a, float b)
                   {
                       return a + b;
                   }
    );
    return res;
}

}// Numeric namespace.
