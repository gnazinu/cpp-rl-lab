#include "cli/command_line.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace rl::cli {

namespace {

std::size_t parse_size_t(const std::string& value, const std::string& flag) {
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + value);
    }
}

int parse_int(const std::string& value, const std::string& flag) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + value);
    }
}

std::uint64_t parse_uint64(const std::string& value, const std::string& flag) {
    try {
        return static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + value);
    }
}

double parse_double(const std::string& value, const std::string& flag) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + value);
    }
}

std::filesystem::path default_output_dir(const CommandMode mode) {
    switch (mode) {
        case CommandMode::Train:
            return "outputs/train";
        case CommandMode::Eval:
            return "outputs/eval";
        case CommandMode::Random:
            return "outputs/random";
        case CommandMode::Help:
            return "outputs";
    }

    return "outputs";
}

CommandMode parse_mode(const std::string& raw_mode) {
    if (raw_mode == "train") {
        return CommandMode::Train;
    }
    if (raw_mode == "eval") {
        return CommandMode::Eval;
    }
    if (raw_mode == "random") {
        return CommandMode::Random;
    }
    if (raw_mode == "help" || raw_mode == "--help" || raw_mode == "-h") {
        return CommandMode::Help;
    }

    throw std::runtime_error("unknown command: " + raw_mode);
}

std::string require_value(
    const int argc,
    char** argv,
    int& index,
    const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
    }
    ++index;
    return argv[index];
}

}  // namespace

CommandLineOptions parse_arguments(const int argc, char** argv) {
    if (argc < 2) {
        return {};
    }

    CommandLineOptions options;
    options.mode = parse_mode(argv[1]);
    if (options.mode == CommandMode::Help) {
        return options;
    }

    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--help" || argument == "-h") {
            options.mode = CommandMode::Help;
            return options;
        }
        if (argument == "--full-action-space") {
            options.random_valid_actions_only = false;
            continue;
        }
        if (argument == "--maze") {
            options.maze_path = require_value(argc, argv, index, argument);
            continue;
        }
        if (argument == "--episodes") {
            options.episodes =
                parse_size_t(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--max-steps") {
            options.max_steps =
                parse_int(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--seed") {
            options.seed =
                parse_uint64(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--output-dir") {
            options.output_dir = require_value(argc, argv, index, argument);
            continue;
        }
        if (argument == "--policy") {
            options.policy_path = require_value(argc, argv, index, argument);
            continue;
        }
        if (argument == "--eval-interval") {
            options.evaluation_interval =
                parse_size_t(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--eval-episodes") {
            options.evaluation_episodes =
                parse_size_t(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--learning-rate") {
            options.learning_rate =
                parse_double(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--discount") {
            options.discount_factor =
                parse_double(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--epsilon-start") {
            options.epsilon_start =
                parse_double(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--epsilon-min") {
            options.epsilon_min =
                parse_double(require_value(argc, argv, index, argument), argument);
            continue;
        }
        if (argument == "--epsilon-decay") {
            options.epsilon_decay =
                parse_double(require_value(argc, argv, index, argument), argument);
            continue;
        }

        throw std::runtime_error("unknown option: " + argument);
    }

    if (options.episodes == 0) {
        throw std::runtime_error("--episodes must be greater than zero");
    }
    if (options.max_steps < 0) {
        throw std::runtime_error("--max-steps must be non-negative");
    }
    if (options.mode == CommandMode::Eval && options.policy_path.empty()) {
        throw std::runtime_error("--policy is required in eval mode");
    }
    if (options.output_dir.empty()) {
        options.output_dir = default_output_dir(options.mode);
    }

    return options;
}

std::string usage(const char* program_name) {
    return std::string("cpp-rl-lab CLI\n\n") +
           "Usage:\n" +
           "  " + program_name +
           " train [--maze <path>] [--episodes <n>] [--max-steps <n>] "
           "[--seed <n>] [--output-dir <dir>]\n" +
           "  " + program_name +
           " eval --policy <path> [--maze <path>] [--episodes <n>] "
           "[--max-steps <n>] [--seed <n>] [--output-dir <dir>]\n" +
           "  " + program_name +
           " random [--maze <path>] [--episodes <n>] [--max-steps <n>] "
           "[--seed <n>] [--output-dir <dir>] [--full-action-space]\n\n" +
           "Key options:\n" +
           "  --maze <path>             Load a maze from a text file. If omitted, "
           "the built-in default maze is used.\n" +
           "  --episodes <n>            Number of train/eval episodes.\n" +
           "  --max-steps <n>           Maximum steps per episode. Defaults to a "
           "maze-size-based heuristic.\n" +
           "  --seed <n>                Seed for deterministic agent behavior.\n" +
           "  --output-dir <dir>        Directory for metrics and checkpoints.\n" +
           "  --policy <path>           Q-table file used by eval mode.\n" +
           "  --eval-interval <n>       Train-mode evaluation cadence.\n" +
           "  --eval-episodes <n>       Number of greedy evaluation episodes.\n" +
           "  --learning-rate <value>   Q-learning alpha.\n" +
           "  --discount <value>        Q-learning gamma.\n" +
           "  --epsilon-start <value>   Initial epsilon.\n" +
           "  --epsilon-min <value>     Minimum epsilon.\n" +
           "  --epsilon-decay <value>   Per-episode epsilon decay.\n\n" +
           "Examples:\n" +
           "  " + program_name +
           " train --maze configs/mazes/basic.txt --episodes 2000 --seed 42\n" +
           "  " + program_name +
           " eval --maze configs/mazes/basic.txt --policy outputs/train/final_policy.qtable --episodes 100\n" +
           "  " + program_name +
           " random --maze configs/mazes/basic.txt --episodes 100\n";
}

}  // namespace rl::cli
