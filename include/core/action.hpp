#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace rl::core {

enum class Action : std::size_t {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
};

inline const std::array<Action, 4>& all_actions() {
    static const std::array<Action, 4> actions{
        Action::Up,
        Action::Down,
        Action::Left,
        Action::Right,
    };
    return actions;
}

inline std::size_t action_count() {
    return all_actions().size();
}

inline std::size_t to_index(const Action action) {
    return static_cast<std::size_t>(action);
}

inline std::string to_string(const Action action) {
    switch (action) {
        case Action::Up:
            return "up";
        case Action::Down:
            return "down";
        case Action::Left:
            return "left";
        case Action::Right:
            return "right";
    }

    return "unknown";
}

}  // namespace rl::core
