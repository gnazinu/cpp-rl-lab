#include "env/maze_layout.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rl::env {

namespace {

char normalize_character(char value) {
    if (value == '\r') {
        return '\0';
    }
    return value;
}

void validate_character(const char cell) {
    switch (cell) {
        case '#':
        case '.':
        case 'S':
        case 'G':
            return;
        default:
            throw std::runtime_error(
                std::string("invalid maze character: '") + cell + "'");
    }
}

}  // namespace

MazeLayout MazeLayout::from_lines(const std::vector<std::string>& lines) {
    if (lines.empty()) {
        throw std::runtime_error("maze must contain at least one row");
    }

    const std::size_t width = lines.front().size();
    if (width == 0) {
        throw std::runtime_error("maze rows must not be empty");
    }

    MazeLayout layout;
    layout.width = static_cast<int>(width);
    layout.height = static_cast<int>(lines.size());
    layout.rows = lines;

    int start_count = 0;
    int goal_count = 0;

    for (std::size_t row = 0; row < lines.size(); ++row) {
        if (lines[row].size() != width) {
            throw std::runtime_error("maze rows must all have the same width");
        }

        for (std::size_t col = 0; col < width; ++col) {
            const char cell = lines[row][col];
            validate_character(cell);

            if (cell == 'S') {
                layout.start = {
                    static_cast<int>(row),
                    static_cast<int>(col),
                };
                ++start_count;
            } else if (cell == 'G') {
                layout.goal = {
                    static_cast<int>(row),
                    static_cast<int>(col),
                };
                ++goal_count;
            }
        }
    }

    if (start_count != 1) {
        throw std::runtime_error("maze must contain exactly one start cell");
    }
    if (goal_count != 1) {
        throw std::runtime_error("maze must contain exactly one goal cell");
    }

    return layout;
}

MazeLayout MazeLayout::from_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open maze file: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        while (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            continue;
        }

        for (char& cell : line) {
            const char normalized = normalize_character(cell);
            if (normalized == '\0') {
                continue;
            }
            cell = normalized;
        }

        lines.push_back(line);
    }

    return from_lines(lines);
}

MazeLayout MazeLayout::default_layout() {
    return from_lines({
        "########",
        "#S.....#",
        "#.###..#",
        "#...#G.#",
        "########",
    });
}

bool MazeLayout::is_in_bounds(const core::Position& position) const {
    return position.row >= 0 && position.row < height && position.col >= 0 &&
           position.col < width;
}

bool MazeLayout::is_wall(const core::Position& position) const {
    if (!is_in_bounds(position)) {
        return true;
    }

    return rows[static_cast<std::size_t>(position.row)]
               [static_cast<std::size_t>(position.col)] == '#';
}

bool MazeLayout::is_walkable(const core::Position& position) const {
    return is_in_bounds(position) && !is_wall(position);
}

int MazeLayout::flatten(const core::Position& position) const {
    if (!is_in_bounds(position)) {
        throw std::runtime_error("cannot flatten out-of-bounds position");
    }

    return position.row * width + position.col;
}

core::Position MazeLayout::unflatten(const int state) const {
    const auto total_cells = static_cast<int>(cell_count());
    if (state < 0 || state >= total_cells) {
        throw std::runtime_error("state index is out of bounds");
    }

    return {
        state / width,
        state % width,
    };
}

std::size_t MazeLayout::cell_count() const {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

std::size_t MazeLayout::walkable_cell_count() const {
    std::size_t count = 0;
    for (const auto& row : rows) {
        for (const char cell : row) {
            if (cell != '#') {
                ++count;
            }
        }
    }
    return count;
}

}  // namespace rl::env
