#pragma once

#include <cstddef>

namespace rl::core {

struct Position {
    int row = 0;
    int col = 0;
};

inline bool operator==(const Position& lhs, const Position& rhs) {
    return lhs.row == rhs.row && lhs.col == rhs.col;
}

inline bool operator!=(const Position& lhs, const Position& rhs) {
    return !(lhs == rhs);
}

struct StepResult {
    int next_state = 0;
    double reward = 0.0;
    bool done = false;
    bool solved = false;
    bool truncated = false;
};

struct EpisodeStats {
    std::size_t episode = 0;
    double total_reward = 0.0;
    std::size_t steps = 0;
    bool solved = false;
    double epsilon = 0.0;
    double moving_average_reward = 0.0;
    double success_rate = 0.0;
};

}  // namespace rl::core
