#include "plot.h"
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#include <filesystem>

void plotTime(std::vector<PerfTestResult> const &info,
              std::string const &output_dir_name)
{
    auto const num_experiments = std::size(info);
    auto data = std::vector<std::vector<float>>{};
    data.reserve(num_experiments);
    auto labels = std::vector<std::string>{};
    labels.reserve(num_experiments);

    for (auto [label, perf_vec]: info) {
        labels.emplace_back(std::move(label));
        data.emplace_back(std::move(perf_vec));
    }

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

void plotPerfBoost(std::vector<PerfTestResult> const &info,
                   std::string const &output_dir_name)
{
    auto const ref_metric = info.front().metric_vals;
    auto const num_experiments = std::size(info);
    auto perf_boost_info = std::vector<PerfTestResult>{};
    perf_boost_info.reserve(num_experiments - 1);

    std::transform(std::begin(info) + 1u, std::end(info), std::back_inserter(perf_boost_info),
                   [&ref_metric](auto const &perf_test)
                   {
                       auto const &[label, perf_vec] = perf_test;
                       auto boost_vec = std::vector<float>{};
                       std::transform(std::begin(perf_vec), std::end(perf_vec), std::begin(ref_metric),
                                      std::back_inserter(boost_vec),
                                      [](auto const &perf, auto const &ref_perf)
                                      {
                                          return (ref_perf - perf) * 100.0f / ref_perf;
                                      }
                       );
                       return PerfTestResult{label, std::move(boost_vec)};
                   }
    );

    auto run_idx = std::vector<float>(std::size(ref_metric));
    std::iota(std::begin(run_idx), std::end(run_idx), 0.0f);


    auto keywords = std::map<std::string, std::string>{};
    keywords["font.size"] = "24";
    keywords["lines.linewidth"] = "5";

    matplotlibcpp::rcparams(keywords);
    matplotlibcpp::figure_size(1200, 780);
    matplotlibcpp::title("Performance boost after optimization (%)");
    matplotlibcpp::xlabel("Run");

    for (auto const &[label, perf_boost]: perf_boost_info) {
        matplotlibcpp::named_plot(label, run_idx, perf_boost);
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