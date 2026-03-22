#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "core/types.hpp"

namespace rl::env {

struct MazeLayout {
    int width = 0;
    int height = 0;
    std::vector<std::string> rows;
    core::Position start;
    core::Position goal;

    static MazeLayout from_lines(const std::vector<std::string>& lines);
    static MazeLayout from_file(const std::filesystem::path& path);
    static MazeLayout default_layout();

    bool is_in_bounds(const core::Position& position) const;
    bool is_wall(const core::Position& position) const;
    bool is_walkable(const core::Position& position) const;
    int flatten(const core::Position& position) const;
    core::Position unflatten(int state) const;
    std::size_t cell_count() const;
    std::size_t walkable_cell_count() const;
};

}  // namespace rl::env
