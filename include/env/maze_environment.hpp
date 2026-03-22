#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/action.hpp"
#include "core/types.hpp"
#include "env/environment.hpp"
#include "env/maze_layout.hpp"

namespace rl::env {

struct MazeRewards {
    double goal_reward = 10.0;
    double step_penalty = -0.1;
    double invalid_move_penalty = -0.75;
};

struct MazeEnvironmentConfig {
    MazeLayout layout;
    int max_steps = 0;
    MazeRewards rewards;
};

class MazeEnvironment : public Environment {
  public:
    explicit MazeEnvironment(MazeEnvironmentConfig config);

    static int recommended_max_steps(const MazeLayout& layout);

    int reset() override;
    core::StepResult step(core::Action action) override;
    bool is_terminal() const override;
    int get_state() const override;
    std::vector<core::Action> get_action_space() const override;
    std::vector<core::Action> get_valid_actions() const override;
    std::size_t state_space_size() const override;
    std::size_t action_space_size() const override;
    void seed(std::uint64_t seed) override;
    std::string render() const override;

    int max_steps() const;
    int steps_taken() const;
    bool solved() const;
    const MazeLayout& layout() const;

  private:
    core::Position next_position(core::Action action) const;

    MazeLayout layout_;
    MazeRewards rewards_;
    int max_steps_ = 0;
    int steps_taken_ = 0;
    bool terminal_ = false;
    bool solved_ = false;
    core::Position current_position_;
    std::uint64_t seed_ = 0;
    std::mt19937 rng_;
};

}  // namespace rl::env
