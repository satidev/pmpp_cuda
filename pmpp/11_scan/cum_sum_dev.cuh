#ifndef CUM_SUM_DEV_CUH
#define CUM_SUM_DEV_CUH

#include <vector>

namespace PMPP
{
enum class ScanAlgorithm
{
    KOGGE_STONE
};
std::vector<float> cumSumDev(std::vector<float> const &vec,
                             ScanAlgorithm algo = ScanAlgorithm::KOGGE_STONE);
}// PMPP namespace.


#endif //CUM_SUM_DEV_CUH

