#pragma once

#include <cstddef>
#include <filesystem>

#include "agents/agent.hpp"
#include "core/types.hpp"
#include "env/environment.hpp"

namespace rl::metrics {
class MetricsLogger;
}

namespace rl::training {

struct TrainerConfig {
    std::size_t episodes = 1000;
    std::size_t evaluation_interval = 100;
    std::size_t evaluation_episodes = 50;
    std::size_t moving_average_window = 100;
    bool checkpoint_best_policy = true;
    std::filesystem::path output_dir = "outputs/train";
};

struct EvaluationSummary {
    double average_reward = 0.0;
    double success_rate = 0.0;
    double average_steps = 0.0;
};

struct TrainingSummary {
    double average_reward = 0.0;
    double success_rate = 0.0;
    double best_evaluation_success_rate = 0.0;
    std::filesystem::path final_policy_path;
    std::filesystem::path best_policy_path;
};

class Trainer {
  public:
    explicit Trainer(TrainerConfig config);

    TrainingSummary train(
        env::Environment& environment,
        agents::Agent& agent,
        metrics::MetricsLogger& logger);
    EvaluationSummary evaluate(
        env::Environment& environment,
        agents::Agent& agent,
        metrics::MetricsLogger* logger = nullptr,
        std::size_t episodes_override = 0);

  private:
    core::EpisodeStats run_episode(
        env::Environment& environment,
        agents::Agent& agent,
        std::size_t episode_number,
        bool training_mode);

    TrainerConfig config_;
};

}  // namespace rl::training
