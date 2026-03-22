#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "agents/q_learning_agent.hpp"
#include "agents/random_agent.hpp"
#include "cli/command_line.hpp"
#include "env/maze_environment.hpp"
#include "env/maze_layout.hpp"
#include "metrics/metrics_logger.hpp"
#include "render/dashboard.hpp"
#include "render/trace_recorder.hpp"
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
    const std::filesystem::path& metrics_path,
    const std::filesystem::path& dashboard_path) {
    std::cout << std::fixed << std::setprecision(3)
              << label << " avg_reward=" << summary.average_reward
              << " success_rate=" << summary.success_rate
              << " avg_steps=" << summary.average_steps
              << " metrics=" << metrics_path.string()
              << " dashboard=" << dashboard_path.string() << '\n';
}

std::string format_double(const double value, const int precision = 3) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::size_t record_every_for_target_count(
    const std::size_t total_episodes,
    const std::size_t target_count) {
    if (target_count == 0 || total_episodes <= target_count) {
        return 1;
    }

    return std::max<std::size_t>(1, total_episodes / target_count);
}

std::vector<std::vector<double>> collect_state_action_values(
    const rl::agents::Agent& agent,
    const rl::env::Environment& environment) {
    std::vector<std::vector<double>> values;
    values.reserve(environment.state_space_size());

    bool has_values = false;
    for (std::size_t state = 0; state < environment.state_space_size(); ++state) {
        auto state_values = agent.action_values(static_cast<int>(state));
        if (!state_values.empty()) {
            has_values = true;
        }
        values.push_back(std::move(state_values));
    }

    if (!has_values) {
        return {};
    }
    return values;
}

std::vector<rl::render::EvaluationPoint> to_render_points(
    const std::vector<rl::training::EvaluationCheckpoint>& history) {
    std::vector<rl::render::EvaluationPoint> points;
    points.reserve(history.size());
    for (const auto& checkpoint : history) {
        points.push_back({
            checkpoint.episode,
            checkpoint.average_reward,
            checkpoint.success_rate,
            checkpoint.average_steps,
        });
    }
    return points;
}

std::vector<rl::render::EpisodeTrace> merge_traces(
    const std::vector<rl::render::EpisodeTrace>& lhs,
    const std::vector<rl::render::EpisodeTrace>& rhs) {
    auto merged = lhs;
    merged.insert(merged.end(), rhs.begin(), rhs.end());
    return merged;
}

rl::render::DashboardField make_field(
    const std::string& label,
    const std::string& value) {
    return {label, value};
}

std::filesystem::path export_dashboard(
    const std::filesystem::path& output_dir,
    rl::render::DashboardData data) {
    return rl::render::export_dashboard_bundle(output_dir, data);
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
        const auto maze_source = options.maze_path.empty() ? "built-in-default"
                                                           : options.maze_path.string();

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
            rl::render::TraceRecorder training_traces({
                "training",
                std::max<std::size_t>(1, options.trace_interval),
                128,
                true,
                true,
            });

            rl::metrics::MetricsLogger logger(
                options.output_dir / "training_metrics.csv",
                trainer_config.moving_average_window);

            std::cout << "Training q_learning episodes=" << options.episodes
                      << " max_steps=" << max_steps << " seed=" << options.seed
                      << " maze="
                      << maze_source
                      << '\n';

            const auto summary =
                trainer.train(environment, agent, logger, &training_traces);
            rl::render::TraceRecorder evaluation_traces({
                "evaluation",
                1,
                std::max<std::size_t>(1, options.dashboard_episodes),
                true,
                true,
            });
            trainer.evaluate(
                environment,
                agent,
                nullptr,
                options.dashboard_episodes,
                &evaluation_traces,
                "evaluation");

            rl::render::DashboardData dashboard_data;
            dashboard_data.title = "cpp-rl-lab Visual Dashboard";
            dashboard_data.subtitle =
                "Interactive playback of training, policy behavior, and maze "
                "metrics.";
            dashboard_data.mode = "train";
            dashboard_data.agent_name = agent.name();
            dashboard_data.maze_source = maze_source;
            dashboard_data.policy_path = summary.final_policy_path.string();
            dashboard_data.best_policy_path = summary.best_policy_path.string();
            dashboard_data.metrics_path = logger.csv_path().string();
            dashboard_data.seed = options.seed;
            dashboard_data.max_steps = max_steps;
            dashboard_data.rewards = {};
            dashboard_data.layout = layout;
            dashboard_data.episode_metrics = logger.records();
            dashboard_data.evaluation_points =
                to_render_points(summary.evaluation_history);
            dashboard_data.traces = merge_traces(
                training_traces.episodes(),
                evaluation_traces.episodes());
            dashboard_data.state_action_values =
                collect_state_action_values(agent, environment);
            dashboard_data.summary_fields = {
                make_field("Avg Reward", format_double(summary.average_reward)),
                make_field("Success Rate", format_double(summary.success_rate)),
                make_field(
                    "Best Eval Success",
                    format_double(summary.best_evaluation_success_rate)),
                make_field(
                    "Final Epsilon",
                    logger.records().empty()
                        ? "0.000"
                        : format_double(logger.records().back().epsilon)),
                make_field(
                    "Traced Episodes",
                    std::to_string(dashboard_data.traces.size())),
                make_field(
                    "Maze Cells",
                    std::to_string(layout.walkable_cell_count())),
            };
            dashboard_data.configuration_fields = {
                make_field("Mode", "train"),
                make_field("Agent", agent.name()),
                make_field("Maze", maze_source),
                make_field("Episodes", std::to_string(options.episodes)),
                make_field("Max Steps", std::to_string(max_steps)),
                make_field("Seed", std::to_string(options.seed)),
                make_field(
                    "Eval Interval",
                    std::to_string(options.evaluation_interval)),
                make_field(
                    "Eval Episodes",
                    std::to_string(options.evaluation_episodes)),
                make_field(
                    "Trace Interval",
                    std::to_string(options.trace_interval)),
                make_field(
                    "Dashboard Episodes",
                    std::to_string(options.dashboard_episodes)),
                make_field(
                    "Learning Rate",
                    format_double(options.learning_rate, 4)),
                make_field(
                    "Discount",
                    format_double(options.discount_factor, 4)),
                make_field(
                    "Epsilon Start",
                    format_double(options.epsilon_start, 4)),
                make_field(
                    "Epsilon Min",
                    format_double(options.epsilon_min, 4)),
                make_field(
                    "Epsilon Decay",
                    format_double(options.epsilon_decay, 4)),
                make_field(
                    "Reward Goal / Step / Invalid",
                    format_double(10.0, 1) + " / " + format_double(-0.1, 1) +
                        " / " + format_double(-0.75, 2)),
                make_field("Metrics CSV", logger.csv_path().string()),
                make_field("Final Policy", summary.final_policy_path.string()),
                make_field("Best Policy", summary.best_policy_path.string()),
            };
            const auto dashboard_path =
                export_dashboard(options.output_dir, std::move(dashboard_data));
            std::cout << std::fixed << std::setprecision(3)
                      << "training_complete avg_reward=" << summary.average_reward
                      << " success_rate=" << summary.success_rate
                      << " best_eval_success_rate="
                      << summary.best_evaluation_success_rate
                      << " final_policy=" << summary.final_policy_path.string()
                      << " best_policy=" << summary.best_policy_path.string()
                      << " metrics=" << logger.csv_path().string()
                      << " dashboard=" << dashboard_path.string() << '\n';
            return 0;
        }

        if (options.mode == rl::cli::CommandMode::Eval) {
            rl::agents::QLearningConfig agent_config;
            agent_config.state_space_size = environment.state_space_size();

            rl::agents::QLearningAgent agent(agent_config);
            agent.seed(options.seed + 1);
            agent.load(options.policy_path);
            rl::render::TraceRecorder evaluation_traces({
                "evaluation",
                record_every_for_target_count(
                    options.episodes,
                    options.dashboard_episodes),
                std::max<std::size_t>(1, options.dashboard_episodes + 2),
                true,
                true,
            });

            rl::metrics::MetricsLogger logger(
                options.output_dir / "evaluation_metrics.csv",
                trainer_config.moving_average_window);
            const auto summary = trainer.evaluate(
                environment,
                agent,
                &logger,
                options.episodes,
                &evaluation_traces,
                "evaluation");

            rl::render::DashboardData dashboard_data;
            dashboard_data.title = "cpp-rl-lab Visual Dashboard";
            dashboard_data.subtitle =
                "Interactive playback of a saved policy solving the maze.";
            dashboard_data.mode = "eval";
            dashboard_data.agent_name = agent.name();
            dashboard_data.maze_source = maze_source;
            dashboard_data.policy_path = options.policy_path.string();
            dashboard_data.metrics_path = logger.csv_path().string();
            dashboard_data.seed = options.seed;
            dashboard_data.max_steps = max_steps;
            dashboard_data.rewards = {};
            dashboard_data.layout = layout;
            dashboard_data.episode_metrics = logger.records();
            dashboard_data.traces = evaluation_traces.episodes();
            dashboard_data.state_action_values =
                collect_state_action_values(agent, environment);
            dashboard_data.summary_fields = {
                make_field("Avg Reward", format_double(summary.average_reward)),
                make_field("Success Rate", format_double(summary.success_rate)),
                make_field("Avg Steps", format_double(summary.average_steps)),
                make_field(
                    "Tracked Episodes",
                    std::to_string(dashboard_data.traces.size())),
            };
            dashboard_data.configuration_fields = {
                make_field("Mode", "eval"),
                make_field("Agent", agent.name()),
                make_field("Maze", maze_source),
                make_field("Policy", options.policy_path.string()),
                make_field("Episodes", std::to_string(options.episodes)),
                make_field("Max Steps", std::to_string(max_steps)),
                make_field("Seed", std::to_string(options.seed)),
                make_field(
                    "Dashboard Episodes",
                    std::to_string(options.dashboard_episodes)),
                make_field("Metrics CSV", logger.csv_path().string()),
            };
            const auto dashboard_path =
                export_dashboard(options.output_dir, std::move(dashboard_data));
            print_evaluation_summary(
                "evaluation_complete",
                summary,
                logger.csv_path(),
                dashboard_path);
            return 0;
        }

        rl::agents::RandomAgent agent(
            {.valid_actions_only = options.random_valid_actions_only});
        agent.seed(options.seed + 1);
        rl::render::TraceRecorder random_traces({
            "random",
            record_every_for_target_count(
                options.episodes,
                options.dashboard_episodes),
            std::max<std::size_t>(1, options.dashboard_episodes + 2),
            true,
            true,
        });

        rl::metrics::MetricsLogger logger(
            options.output_dir / "random_metrics.csv",
            trainer_config.moving_average_window);
        const auto summary = trainer.evaluate(
            environment,
            agent,
            &logger,
            options.episodes,
            &random_traces,
            "random");
        rl::render::DashboardData dashboard_data;
        dashboard_data.title = "cpp-rl-lab Visual Dashboard";
        dashboard_data.subtitle =
            "Interactive baseline playback for comparing random exploration "
            "against learned policies.";
        dashboard_data.mode = "random";
        dashboard_data.agent_name = agent.name();
        dashboard_data.maze_source = maze_source;
        dashboard_data.metrics_path = logger.csv_path().string();
        dashboard_data.seed = options.seed;
        dashboard_data.max_steps = max_steps;
        dashboard_data.rewards = {};
        dashboard_data.layout = layout;
        dashboard_data.episode_metrics = logger.records();
        dashboard_data.traces = random_traces.episodes();
        dashboard_data.summary_fields = {
            make_field("Avg Reward", format_double(summary.average_reward)),
            make_field("Success Rate", format_double(summary.success_rate)),
            make_field("Avg Steps", format_double(summary.average_steps)),
            make_field(
                "Tracked Episodes",
                std::to_string(dashboard_data.traces.size())),
        };
        dashboard_data.configuration_fields = {
            make_field("Mode", "random"),
            make_field("Agent", agent.name()),
            make_field("Maze", maze_source),
            make_field("Episodes", std::to_string(options.episodes)),
            make_field("Max Steps", std::to_string(max_steps)),
            make_field("Seed", std::to_string(options.seed)),
            make_field(
                "Action Sampling",
                options.random_valid_actions_only ? "valid actions only"
                                                  : "full action space"),
            make_field(
                "Dashboard Episodes",
                std::to_string(options.dashboard_episodes)),
            make_field("Metrics CSV", logger.csv_path().string()),
        };
        const auto dashboard_path =
            export_dashboard(options.output_dir, std::move(dashboard_data));
        print_evaluation_summary(
            "random_complete",
            summary,
            logger.csv_path(),
            dashboard_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
