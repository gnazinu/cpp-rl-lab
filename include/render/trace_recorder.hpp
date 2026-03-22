#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/action.hpp"
#include "core/types.hpp"

namespace rl::render {

struct StepTrace {
    std::size_t index = 0;
    int state = 0;
    int next_state = 0;
    std::string action;
    double reward = 0.0;
    double cumulative_reward = 0.0;
    bool done = false;
    bool solved = false;
    bool truncated = false;
    bool blocked = false;
    std::vector<std::string> valid_actions;
    std::vector<double> action_values;
};

struct EpisodeTrace {
    std::size_t episode = 0;
    std::string phase;
    bool training_mode = false;
    int initial_state = 0;
    double total_reward = 0.0;
    std::size_t steps = 0;
    bool solved = false;
    double epsilon = 0.0;
    std::vector<StepTrace> steps_trace;
};

struct TraceRecorderConfig {
    std::string phase = "training";
    std::size_t record_every = 100;
    std::size_t max_episodes = 128;
    bool record_first_episode = true;
    bool record_last_episode = true;
};

class TraceRecorder {
  public:
    explicit TraceRecorder(TraceRecorderConfig config);

    bool should_record_episode(
        std::size_t episode_number,
        std::size_t total_episodes) const;
    void begin_episode(
        std::size_t episode_number,
        const std::string& phase,
        bool training_mode,
        int initial_state,
        double epsilon);
    void record_step(
        std::size_t step_index,
        int state,
        core::Action action,
        const core::StepResult& result,
        const std::vector<core::Action>& valid_actions,
        const std::vector<double>& action_values);
    void end_episode(const core::EpisodeStats& stats);

    const std::vector<EpisodeTrace>& episodes() const;

  private:
    TraceRecorderConfig config_;
    std::vector<EpisodeTrace> episodes_;
    bool is_recording_ = false;
    double cumulative_reward_ = 0.0;
};

}  // namespace rl::render
