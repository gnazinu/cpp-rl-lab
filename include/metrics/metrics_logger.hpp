#pragma once

#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/types.hpp"

namespace rl::metrics {

class MetricsLogger {
  public:
    explicit MetricsLogger(
        std::filesystem::path csv_path,
        std::size_t moving_average_window = 100);

    core::EpisodeStats log_episode(const core::EpisodeStats& stats);
    const std::vector<core::EpisodeStats>& records() const;
    const std::filesystem::path& csv_path() const;

  private:
    std::filesystem::path csv_path_;
    std::ofstream stream_;
    std::size_t moving_average_window_ = 100;
    std::deque<double> reward_window_;
    double reward_window_sum_ = 0.0;
    std::size_t success_count_ = 0;
    std::vector<core::EpisodeStats> records_;
};

}  // namespace rl::metrics
