#ifndef CUM_SUM_HOST_H
#define CUM_SUM_HOST_H

#include <vector>

namespace PMPP
{
// Cumulative sum of a set of numbers.
// Inclusive scan with and addition operator.
std::vector<float> cumSumHost(std::vector<float> const &vec);
}

#endif //CUM_SUM_HOST_H
