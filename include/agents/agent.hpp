#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/action.hpp"

namespace rl::agents {

class Agent {
  public:
    virtual ~Agent() = default;

    virtual std::string name() const = 0;
    virtual void seed(std::uint64_t seed) = 0;
    virtual void begin_episode() {}
    virtual core::Action select_action(
        int state,
        const std::vector<core::Action>& available_actions,
        bool training_mode) = 0;
    virtual void observe_transition(
        int state,
        core::Action action,
        double reward,
        int next_state,
        bool done,
        const std::vector<core::Action>& next_available_actions) = 0;
    virtual void end_episode() {}
    virtual void save(const std::filesystem::path& path) const = 0;
    virtual void load(const std::filesystem::path& path) = 0;
    virtual double current_epsilon() const {
        return 0.0;
    }
    virtual std::vector<double> action_values(int state) const {
        (void)state;
        return {};
    }
};

}  // namespace rl::agents
