#include "render/trace_recorder.hpp"

#include <utility>

namespace rl::render {

TraceRecorder::TraceRecorder(TraceRecorderConfig config)
    : config_(std::move(config)) {}

bool TraceRecorder::should_record_episode(
    const std::size_t episode_number,
    const std::size_t total_episodes) const {
    if (config_.max_episodes == 0 || episodes_.size() >= config_.max_episodes) {
        return false;
    }

    if (config_.record_first_episode && episode_number == 1) {
        return true;
    }
    if (config_.record_last_episode && episode_number == total_episodes) {
        return true;
    }
    return config_.record_every > 0 && episode_number % config_.record_every == 0;
}

void TraceRecorder::begin_episode(
    const std::size_t episode_number,
    const std::string& phase,
    const bool training_mode,
    const int initial_state,
    const double epsilon) {
    if (episodes_.size() >= config_.max_episodes) {
        is_recording_ = false;
        return;
    }

    episodes_.push_back({
        episode_number,
        phase.empty() ? config_.phase : phase,
        training_mode,
        initial_state,
        0.0,
        0,
        false,
        epsilon,
        {},
    });
    cumulative_reward_ = 0.0;
    is_recording_ = true;
}

void TraceRecorder::record_step(
    const std::size_t step_index,
    const int state,
    const core::Action action,
    const core::StepResult& result,
    const std::vector<core::Action>& valid_actions,
    const std::vector<double>& action_values) {
    if (!is_recording_ || episodes_.empty()) {
        return;
    }

    cumulative_reward_ += result.reward;

    std::vector<std::string> valid_action_names;
    valid_action_names.reserve(valid_actions.size());
    for (const auto valid_action : valid_actions) {
        valid_action_names.push_back(core::to_string(valid_action));
    }

    episodes_.back().steps_trace.push_back({
        step_index,
        state,
        result.next_state,
        core::to_string(action),
        result.reward,
        cumulative_reward_,
        result.done,
        result.solved,
        result.truncated,
        state == result.next_state && !result.solved,
        std::move(valid_action_names),
        action_values,
    });
}

void TraceRecorder::end_episode(const core::EpisodeStats& stats) {
    if (!is_recording_ || episodes_.empty()) {
        return;
    }

    auto& episode = episodes_.back();
    episode.total_reward = stats.total_reward;
    episode.steps = stats.steps;
    episode.solved = stats.solved;
    episode.epsilon = stats.epsilon;
    is_recording_ = false;
}

const std::vector<EpisodeTrace>& TraceRecorder::episodes() const {
    return episodes_;
}

}  // namespace rl::render
