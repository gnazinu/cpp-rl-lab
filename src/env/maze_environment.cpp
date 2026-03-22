#include "env/maze_environment.hpp"

#include <sstream>
#include <stdexcept>

namespace rl::env {

MazeEnvironment::MazeEnvironment(MazeEnvironmentConfig config)
    : layout_(std::move(config.layout)),
      rewards_(config.rewards),
      max_steps_(config.max_steps > 0 ? config.max_steps
                                      : recommended_max_steps(layout_)),
      current_position_(layout_.start),
      rng_(std::random_device{}()) {
    if (max_steps_ <= 0) {
        throw std::runtime_error("max_steps must be positive");
    }
    if (!layout_.is_walkable(layout_.start)) {
        throw std::runtime_error("maze start must be walkable");
    }
    if (!layout_.is_walkable(layout_.goal)) {
        throw std::runtime_error("maze goal must be walkable");
    }
}

int MazeEnvironment::recommended_max_steps(const MazeLayout& layout) {
    return std::max(32, layout.width * layout.height * 4);
}

int MazeEnvironment::reset() {
    current_position_ = layout_.start;
    steps_taken_ = 0;
    terminal_ = false;
    solved_ = false;
    return get_state();
}

core::StepResult MazeEnvironment::step(const core::Action action) {
    if (terminal_) {
        throw std::runtime_error("cannot step a terminal environment");
    }

    ++steps_taken_;
    double reward = rewards_.step_penalty;

    const auto candidate = next_position(action);
    if (layout_.is_walkable(candidate)) {
        current_position_ = candidate;
    } else {
        reward += rewards_.invalid_move_penalty;
    }

    bool truncated = false;
    if (current_position_ == layout_.goal) {
        solved_ = true;
        terminal_ = true;
        reward += rewards_.goal_reward;
    } else if (steps_taken_ >= max_steps_) {
        terminal_ = true;
        truncated = true;
    }

    return {
        get_state(),
        reward,
        terminal_,
        solved_,
        truncated,
    };
}

bool MazeEnvironment::is_terminal() const {
    return terminal_;
}

int MazeEnvironment::get_state() const {
    return layout_.flatten(current_position_);
}

std::vector<core::Action> MazeEnvironment::get_action_space() const {
    const auto& actions = core::all_actions();
    return {actions.begin(), actions.end()};
}

std::vector<core::Action> MazeEnvironment::get_valid_actions() const {
    std::vector<core::Action> actions;
    for (const auto action : core::all_actions()) {
        if (layout_.is_walkable(next_position(action))) {
            actions.push_back(action);
        }
    }
    return actions;
}

std::size_t MazeEnvironment::state_space_size() const {
    return layout_.cell_count();
}

std::size_t MazeEnvironment::action_space_size() const {
    return core::action_count();
}

void MazeEnvironment::seed(const std::uint64_t seed) {
    seed_ = seed;
    rng_.seed(static_cast<std::mt19937::result_type>(seed_));
}

std::string MazeEnvironment::render() const {
    auto view = layout_.rows;
    view[static_cast<std::size_t>(current_position_.row)]
        [static_cast<std::size_t>(current_position_.col)] = 'A';

    std::ostringstream output;
    for (std::size_t row = 0; row < view.size(); ++row) {
        output << view[row];
        if (row + 1 < view.size()) {
            output << '\n';
        }
    }
    return output.str();
}

int MazeEnvironment::max_steps() const {
    return max_steps_;
}

int MazeEnvironment::steps_taken() const {
    return steps_taken_;
}

bool MazeEnvironment::solved() const {
    return solved_;
}

const MazeLayout& MazeEnvironment::layout() const {
    return layout_;
}

core::Position MazeEnvironment::next_position(const core::Action action) const {
    auto next = current_position_;
    switch (action) {
        case core::Action::Up:
            --next.row;
            break;
        case core::Action::Down:
            ++next.row;
            break;
        case core::Action::Left:
            --next.col;
            break;
        case core::Action::Right:
            ++next.col;
            break;
    }
    return next;
}

}  // namespace rl::env
