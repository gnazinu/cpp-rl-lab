#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace rl::cli {

enum class CommandMode {
    Help,
    Train,
    Eval,
    Random,
};

struct CommandLineOptions {
    CommandMode mode = CommandMode::Help;
    std::filesystem::path maze_path;
    std::filesystem::path output_dir;
    std::filesystem::path policy_path;
    std::size_t episodes = 1000;
    int max_steps = 0;
    std::uint64_t seed = 42;
    std::size_t evaluation_interval = 100;
    std::size_t evaluation_episodes = 50;
    double learning_rate = 0.1;
    double discount_factor = 0.95;
    double epsilon_start = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.995;
    std::size_t trace_interval = 100;
    std::size_t dashboard_episodes = 5;
    bool random_valid_actions_only = true;
};

CommandLineOptions parse_arguments(int argc, char** argv);
std::string usage(const char* program_name);

}  // namespace rl::cli
