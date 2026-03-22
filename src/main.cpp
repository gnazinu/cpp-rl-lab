#include <iomanip>
#include <iostream>

#include "agents/q_learning_agent.hpp"
#include "agents/random_agent.hpp"
#include "cli/command_line.hpp"
#include "env/maze_environment.hpp"
#include "env/maze_layout.hpp"
#include "metrics/metrics_logger.hpp"
#include "training/trainer.hpp"
#include "utils/filesystem.hpp"

namespace {

rl::env::MazeLayout load_layout(const rl::cli::CommandLineOptions& options) {
    if (!options.maze_path.empty()) {
        return rl::env::MazeLayout::from_file(options.maze_path);
    }
    return rl::env::MazeLayout::default_layout();
}

int resolve_max_steps(
    const rl::cli::CommandLineOptions& options,
    const rl::env::MazeLayout& layout) {
    if (options.max_steps > 0) {
        return options.max_steps;
    }
    return rl::env::MazeEnvironment::recommended_max_steps(layout);
}

void print_evaluation_summary(
    const std::string& label,
    const rl::training::EvaluationSummary& summary,
    const std::filesystem::path& metrics_path) {
    std::cout << std::fixed << std::setprecision(3)
              << label << " avg_reward=" << summary.average_reward
              << " success_rate=" << summary.success_rate
              << " avg_steps=" << summary.average_steps
              << " metrics=" << metrics_path.string() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = rl::cli::parse_arguments(argc, argv);
        if (options.mode == rl::cli::CommandMode::Help) {
            std::cout << rl::cli::usage(argv[0]);
            return 0;
        }

        const auto layout = load_layout(options);
        const int max_steps = resolve_max_steps(options, layout);
        rl::env::MazeEnvironment environment({layout, max_steps, {}});
        environment.seed(options.seed);

        rl::training::TrainerConfig trainer_config;
        trainer_config.episodes = options.episodes;
        trainer_config.evaluation_interval = options.evaluation_interval;
        trainer_config.evaluation_episodes = options.evaluation_episodes;
        trainer_config.output_dir = options.output_dir;

        rl::training::Trainer trainer(trainer_config);
        rl::utils::ensure_directory(options.output_dir);

        if (options.mode == rl::cli::CommandMode::Train) {
            rl::agents::QLearningConfig agent_config;
            agent_config.state_space_size = environment.state_space_size();
            agent_config.learning_rate = options.learning_rate;
            agent_config.discount_factor = options.discount_factor;
            agent_config.epsilon_start = options.epsilon_start;
            agent_config.epsilon_min = options.epsilon_min;
            agent_config.epsilon_decay = options.epsilon_decay;

            rl::agents::QLearningAgent agent(agent_config);
            agent.seed(options.seed + 1);

            rl::metrics::MetricsLogger logger(
                options.output_dir / "training_metrics.csv",
                trainer_config.moving_average_window);

            std::cout << "Training q_learning episodes=" << options.episodes
                      << " max_steps=" << max_steps << " seed=" << options.seed
                      << " maze="
                      << (options.maze_path.empty() ? "built-in-default"
                                                    : options.maze_path.string())
                      << '\n';

            const auto summary = trainer.train(environment, agent, logger);
            std::cout << std::fixed << std::setprecision(3)
                      << "training_complete avg_reward=" << summary.average_reward
                      << " success_rate=" << summary.success_rate
                      << " best_eval_success_rate="
                      << summary.best_evaluation_success_rate
                      << " final_policy=" << summary.final_policy_path.string()
                      << " best_policy=" << summary.best_policy_path.string()
                      << " metrics=" << logger.csv_path().string() << '\n';
            return 0;
        }

        if (options.mode == rl::cli::CommandMode::Eval) {
            rl::agents::QLearningConfig agent_config;
            agent_config.state_space_size = environment.state_space_size();

            rl::agents::QLearningAgent agent(agent_config);
            agent.seed(options.seed + 1);
            agent.load(options.policy_path);

            rl::metrics::MetricsLogger logger(
                options.output_dir / "evaluation_metrics.csv",
                trainer_config.moving_average_window);
            const auto summary =
                trainer.evaluate(environment, agent, &logger, options.episodes);
            print_evaluation_summary("evaluation_complete", summary, logger.csv_path());
            return 0;
        }

        rl::agents::RandomAgent agent(
            {.valid_actions_only = options.random_valid_actions_only});
        agent.seed(options.seed + 1);

        rl::metrics::MetricsLogger logger(
            options.output_dir / "random_metrics.csv",
            trainer_config.moving_average_window);
        const auto summary =
            trainer.evaluate(environment, agent, &logger, options.episodes);
        print_evaluation_summary("random_complete", summary, logger.csv_path());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
