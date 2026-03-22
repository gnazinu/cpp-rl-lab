#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "agents/agent.hpp"

namespace rl::agents {

struct RandomAgentConfig {
    bool valid_actions_only = true;
};

class RandomAgent : public Agent {
  public:
    explicit RandomAgent(RandomAgentConfig config = {});

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
    void save(const std::filesystem::path& path) const override;
    void load(const std::filesystem::path& path) override;

  private:
    RandomAgentConfig config_;
    std::mt19937 rng_;
};

}  // namespace rl::agents
