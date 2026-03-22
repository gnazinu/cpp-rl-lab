#include "metrics/metrics_logger.hpp"

#include <iomanip>
#include <stdexcept>

#include "utils/filesystem.hpp"

namespace rl::metrics {

MetricsLogger::MetricsLogger(
    std::filesystem::path csv_path,
    const std::size_t moving_average_window)
    : csv_path_(std::move(csv_path)),
      moving_average_window_(moving_average_window == 0 ? 1
                                                        : moving_average_window) {
    if (!csv_path_.parent_path().empty()) {
        utils::ensure_directory(csv_path_.parent_path());
    }

    stream_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!stream_) {
        throw std::runtime_error(
            "failed to open metrics csv: " + csv_path_.string());
    }

    stream_ << "episode,total_reward,steps,solved,epsilon,"
               "moving_average_reward,success_rate\n";
    stream_ << std::fixed << std::setprecision(6);
}

core::EpisodeStats MetricsLogger::log_episode(const core::EpisodeStats& stats) {
    reward_window_.push_back(stats.total_reward);
    reward_window_sum_ += stats.total_reward;
    if (reward_window_.size() > moving_average_window_) {
        reward_window_sum_ -= reward_window_.front();
        reward_window_.pop_front();
    }

    if (stats.solved) {
        ++success_count_;
    }

    auto enriched = stats;
    enriched.moving_average_reward =
        reward_window_sum_ / static_cast<double>(reward_window_.size());
    enriched.success_rate =
        static_cast<double>(success_count_) /
        static_cast<double>(records_.size() + 1);

    records_.push_back(enriched);
    stream_ << enriched.episode << ',' << enriched.total_reward << ','
            << enriched.steps << ',' << (enriched.solved ? 1 : 0) << ','
            << enriched.epsilon << ',' << enriched.moving_average_reward << ','
            << enriched.success_rate << '\n';
    stream_.flush();

    return enriched;
}

const std::vector<core::EpisodeStats>& MetricsLogger::records() const {
    return records_;
}

const std::filesystem::path& MetricsLogger::csv_path() const {
    return csv_path_;
}

}  // namespace rl::metrics
