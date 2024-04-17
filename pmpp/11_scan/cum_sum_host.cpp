#include <numeric>
#include "cum_sum_host.h"

namespace PMPP
{
std::vector<float> cumSumHost(std::vector<float> const &vec)
{
    if(std::empty(vec))
        return std::vector<float>{};

    auto res = std::vector<float>{};
    res.reserve(std::size(vec));

    std::inclusive_scan(std::begin(vec), std::end(vec), std::back_inserter(res),
                        std::plus<float>{});
    return res;
}
}// PMPP namespace.