#ifndef PLOT_H
#define PLOT_H

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include "utils/perf.cuh"

void plotTime(std::vector<PerfTestResult> const &info, std::string const &output_dir_name);
void plotPerfBoost(std::vector<PerfTestResult> const &info, std::string const &output_dir_name);

#endif //PLOT_H

