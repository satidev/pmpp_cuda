#include "plot.h"
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#include <filesystem>

void plotTime(std::vector<std::tuple<std::string, std::vector<PerfInfo>>> const &info,
              std::string const &output_dir_name)
{
    auto data = std::vector<std::vector<float>>{};
    auto labels = std::vector<std::string>{};
    for (auto const &[name, perf_vec]: info) {
        auto perf_data = std::vector<float>{};
        for (auto const &perf: perf_vec) {
            perf_data.push_back(perf.kernel_duration_ms);
        }
        data.push_back(perf_data);
        labels.push_back(name);

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
    else{
        auto const output_file = std::filesystem::path{output_dir_name} / "time.png";
        std::cout << "Saving plot to " << output_file << std::endl;
        matplotlibcpp::save(output_file.string());
    }
}

void plotPerfBoost(std::vector<std::tuple<std::string, std::vector<PerfInfo>>> const &info,
                   std::string const &output_dir_name)
{
    auto const ref_time = std::get<1>(info.front());
    auto const num_experiments = std::size(info);
    auto data = std::vector<std::vector<float>>{};

    for (auto exp_idx = 1u; exp_idx < num_experiments; ++exp_idx) {
        auto const &[name, perf_vec] = info[exp_idx];
        auto boost_vec = std::vector<float>{};
        std::transform(std::begin(perf_vec), std::end(perf_vec), std::begin(ref_time),
                       std::back_inserter(boost_vec),
                       [](auto const &perf, auto const &ref_perf)
                       {
                           return (ref_perf.kernel_duration_ms - perf.kernel_duration_ms) * 100.0f /
                               ref_perf.kernel_duration_ms;
                       }
        );
        data.push_back(boost_vec);
    }

    auto run_idx = std::vector<float>(std::size(ref_time));
    std::iota(std::begin(run_idx), std::end(run_idx), 0.0f);


    auto keywords = std::map<std::string, std::string>{};
    keywords["font.size"] = "24";
    keywords["lines.linewidth"] = "5";

    matplotlibcpp::rcparams(keywords);
    matplotlibcpp::figure_size(1200, 780);
    matplotlibcpp::title("Performance boost after optimization (%)");
    matplotlibcpp::xlabel("Run");

    for(auto data_idx = 0u; data_idx < std::size(data); ++data_idx) {
        matplotlibcpp::named_plot(std::get<0>(info[data_idx + 1]), run_idx, data[data_idx]);
    }
    matplotlibcpp::legend();

    // If the output directory does not exist, show the plot
    if (!std::filesystem::exists(output_dir_name)) {
        matplotlibcpp::show();
    }
    else{
        auto const output_file = std::filesystem::path{output_dir_name} / "boost.png";
        std::cout << "Saving plot to " << output_file << std::endl;
        matplotlibcpp::save(output_file.string());
    }
}