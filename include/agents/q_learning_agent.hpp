#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "agents/agent.hpp"

namespace rl::agents {

struct QLearningConfig {
    std::size_t state_space_size = 0;
    double learning_rate = 0.1;
    double discount_factor = 0.95;
    double epsilon_start = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.995;
};

class QLearningAgent : public Agent {
  public:
    explicit QLearningAgent(QLearningConfig config);

    std::string name() const override;
    void seed(std::uint64_t seed) override;
    core::Action select_action(
        int state,
        const std::vector<core::Action>& available_actions,
        bool training_mode) override;
    void observe_transition(
        int state,
        core::Action action,
        double reward,
        int next_state,
        bool done,
        const std::vector<core::Action>& next_available_actions) override;
    void end_episode() override;
    void save(const std::filesystem::path& path) const override;
    void load(const std::filesystem::path& path) override;
    double current_epsilon() const override;
    std::vector<double> action_values(int state) const override;

    double q_value(int state, core::Action action) const;
    void set_q_value(int state, core::Action action, double value);
    std::size_t state_space_size() const;

  private:
    void validate_state(int state) const;
    std::vector<core::Action> normalized_actions(
        const std::vector<core::Action>& available_actions) const;

    QLearningConfig config_;
    std::vector<std::vector<double>> q_table_;
    double epsilon_ = 0.0;
    std::mt19937 rng_;
};

}  // namespace rl::agents
