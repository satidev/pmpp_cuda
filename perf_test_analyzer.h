#ifndef PERF_TEST_ANALYZER_H
#define PERF_TEST_ANALYZER_H

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include "utils/perf.cuh"

class PerfTestAnalyzer
{
private:
    PerfTestResult info_;
public:
    explicit PerfTestAnalyzer(PerfTestResult info)
        :
        info_{std::move(info)}
    {
        setPlotParams();
    }
    void plotPerfMetric(std::string const &output_dir_name,
                        std::string const &title,
                        std::string const &ylabel) const;
    void plotPerfBoostInfo(std::string const &output_dir_name,
                           std::string const & ref_impl) const;

private:
    struct BoxPlotData
    {
        std::vector<std::string> labels;
        std::vector<std::vector<float>> data;
    };
    BoxPlotData getBoxPlotData() const;
    PerfTestResult getPerfBoostInfo(std::string const &ref_impl) const;
    void setPlotParams() const;
};

#endif //PERF_TEST_ANALYZER_H

