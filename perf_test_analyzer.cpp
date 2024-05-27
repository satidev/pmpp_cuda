#include "perf_test_analyzer.h"
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#include <filesystem>

void PerfTestAnalyzer::plotPerfMetric(std::string const &output_dir_name) const
{
    auto const [labels, data] = getBoxPlotData();

    auto keywords = std::map<std::string, std::string>{};
    keywords["font.size"] = "24";
    keywords["boxplot.boxprops.linewidth"] = "3";
    keywords["boxplot.meanprops.linewidth"] = "3";
    keywords["boxplot.medianprops.color"] = "red";
    keywords["boxplot.medianprops.linewidth"] = "3";

    matplotlibcpp::rcparams(keywords);
    matplotlibcpp::figure_size(1200, 780);
    matplotlibcpp::title("Kernel execution time");
    matplotlibcpp::ylabel("Time (ms)");
    matplotlibcpp::boxplot(data, labels);

    // If the output directory does not exist, show the plot
    if (!std::filesystem::exists(output_dir_name)) {
        matplotlibcpp::show();
    }
    else {
        auto const output_file = std::filesystem::path{output_dir_name} / "time.png";
        std::cout << "Saving plot to " << output_file << std::endl;
        matplotlibcpp::save(output_file.string());
    }
}
void PerfTestAnalyzer::plotPerfBoostInfo(std::string const &output_dir_name,
                                         std::string const & ref_impl) const
{
    auto const perf_boost_info = getPerfBoostInfo(ref_impl);

    auto keywords = std::map<std::string, std::string>{};
    keywords["font.size"] = "24";
    keywords["lines.linewidth"] = "5";

    matplotlibcpp::rcparams(keywords);
    matplotlibcpp::figure_size(1200, 780);
    matplotlibcpp::title("Performance boost after optimization (%)");
    matplotlibcpp::xlabel("Run");

    for (auto const &[label, perf_boost]: perf_boost_info) {
        matplotlibcpp::named_plot(label, perf_boost);
    }
    matplotlibcpp::legend();

    // If the output directory does not exist, show the plot
    if (!std::filesystem::exists(output_dir_name)) {
        matplotlibcpp::show();
    }
    else {
        auto const output_file = std::filesystem::path{output_dir_name} / "boost.png";
        std::cout << "Saving plot to " << output_file << std::endl;
        matplotlibcpp::save(output_file.string());
    }
}
PerfTestAnalyzer::BoxPlotData PerfTestAnalyzer::getBoxPlotData() const
{
    auto const num_experiments = std::size(info_);
    auto data = std::vector<std::vector<float>>{};
    data.reserve(num_experiments);
    auto labels = std::vector<std::string>{};
    labels.reserve(num_experiments);

    for (auto const &[label, perf_vec]: info_) {
        labels.emplace_back(label);
        data.emplace_back(perf_vec);
    }
    return PerfTestAnalyzer::BoxPlotData{labels, data};
}
PerfTestResult PerfTestAnalyzer::getPerfBoostInfo(std::string const &ref_impl) const
{
    auto const& ref_metric = info_.at(ref_impl);
    auto perf_boost_info = PerfTestResult{};

    for (auto const &[label, perf_vec]: info_) {
        if (label == ref_impl) {
            continue;
        }
        auto boost_vec = std::vector<float>{};
        boost_vec.reserve(std::size(perf_vec));
        std::transform(std::begin(perf_vec), std::end(perf_vec), std::begin(ref_metric),
                       std::back_inserter(boost_vec),
                       [](auto const &perf, auto const &ref_perf)
                       {
                           return (ref_perf - perf) * 100.0f / ref_perf;
                       }
        );
        perf_boost_info[label] =  std::move(boost_vec);
    }
    return perf_boost_info;
}

