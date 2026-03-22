#include "training/trainer.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

#include "metrics/metrics_logger.hpp"
#include "render/trace_recorder.hpp"
#include "utils/filesystem.hpp"

namespace rl::training {

namespace {

void print_training_progress(
    const core::EpisodeStats& latest,
    const EvaluationSummary& evaluation_summary) {
    std::cout << std::fixed << std::setprecision(3)
              << "[train] episode=" << latest.episode
              << " reward=" << latest.total_reward
              << " moving_avg=" << latest.moving_average_reward
              << " success_rate=" << latest.success_rate
              << " epsilon=" << latest.epsilon
              << " eval_success_rate=" << evaluation_summary.success_rate
              << " eval_avg_reward=" << evaluation_summary.average_reward
              << '\n';
}

}  // namespace

Trainer::Trainer(TrainerConfig config) : config_(std::move(config)) {}

TrainingSummary Trainer::train(
    env::Environment& environment,
    agents::Agent& agent,
    metrics::MetricsLogger& logger,
    render::TraceRecorder* trace_recorder) {
    utils::ensure_directory(config_.output_dir);

    double reward_sum = 0.0;
    std::size_t success_count = 0;
    double best_evaluation_success_rate = -1.0;
    std::vector<EvaluationCheckpoint> evaluation_history;
    const auto final_policy_path = config_.output_dir / "final_policy.qtable";
    const auto best_policy_path = config_.output_dir / "best_policy.qtable";

    for (std::size_t episode = 1; episode <= config_.episodes; ++episode) {
        const auto stats = run_episode(
            environment,
            agent,
            episode,
            true,
            config_.episodes,
            "training",
            trace_recorder);
        const auto logged = logger.log_episode(stats);

        reward_sum += logged.total_reward;
        if (logged.solved) {
            ++success_count;
        }

        const bool should_evaluate =
            (episode == config_.episodes) ||
            (config_.evaluation_interval > 0 &&
             episode % config_.evaluation_interval == 0);

        if (should_evaluate) {
            const auto evaluation_summary = evaluate(
                environment,
                agent,
                nullptr,
                config_.evaluation_episodes,
                nullptr,
                "evaluation");
            print_training_progress(logged, evaluation_summary);
            evaluation_history.push_back({
                episode,
                evaluation_summary.average_reward,
                evaluation_summary.success_rate,
                evaluation_summary.average_steps,
            });

            if (config_.checkpoint_best_policy &&
                evaluation_summary.success_rate > best_evaluation_success_rate) {
                best_evaluation_success_rate = evaluation_summary.success_rate;
                agent.save(best_policy_path);
            }
        }
    }

    agent.save(final_policy_path);

    return {
        reward_sum / static_cast<double>(config_.episodes),
        static_cast<double>(success_count) /
            static_cast<double>(config_.episodes),
        best_evaluation_success_rate < 0.0 ? 0.0
                                           : best_evaluation_success_rate,
        final_policy_path,
        config_.checkpoint_best_policy ? best_policy_path
                                       : std::filesystem::path{},
        std::move(evaluation_history),
    };
}

EvaluationSummary Trainer::evaluate(
    env::Environment& environment,
    agents::Agent& agent,
    metrics::MetricsLogger* logger,
    const std::size_t episodes_override,
    render::TraceRecorder* trace_recorder,
    const std::string& phase) {
    const auto episodes =
        episodes_override > 0 ? episodes_override : config_.evaluation_episodes;

    double reward_sum = 0.0;
    double step_sum = 0.0;
    std::size_t success_count = 0;

    for (std::size_t episode = 1; episode <= episodes; ++episode) {
        auto stats = run_episode(
            environment,
            agent,
            episode,
            false,
            episodes,
            phase,
            trace_recorder);
        stats.epsilon = agent.current_epsilon();

        reward_sum += stats.total_reward;
        step_sum += static_cast<double>(stats.steps);
        if (stats.solved) {
            ++success_count;
        }

        if (logger != nullptr) {
            logger->log_episode(stats);
        }
    }

    return {
        reward_sum / static_cast<double>(episodes),
        static_cast<double>(success_count) / static_cast<double>(episodes),
        step_sum / static_cast<double>(episodes),
    };
}

core::EpisodeStats Trainer::run_episode(
    env::Environment& environment,
    agents::Agent& agent,
    const std::size_t episode_number,
    const bool training_mode,
    const std::size_t total_episodes,
    const std::string& phase,
    render::TraceRecorder* trace_recorder) {
    int state = environment.reset();
    agent.begin_episode();

    double total_reward = 0.0;
    std::size_t steps = 0;
    bool solved = false;
    const bool record_trace =
        trace_recorder != nullptr &&
        trace_recorder->should_record_episode(episode_number, total_episodes);

    if (record_trace) {
        trace_recorder->begin_episode(
            episode_number,
            phase,
            training_mode,
            state,
            agent.current_epsilon());
    }

    while (!environment.is_terminal()) {
        auto available_actions = environment.get_valid_actions();
        if (available_actions.empty()) {
            available_actions = environment.get_action_space();
        }

        const auto action =
            agent.select_action(state, available_actions, training_mode);
        const auto action_values = agent.action_values(state);
        const auto result = environment.step(action);
        ++steps;
        total_reward += result.reward;
        solved = result.solved;

        if (record_trace) {
            trace_recorder->record_step(
                steps,
                state,
                action,
                result,
                available_actions,
                action_values);
        }

        if (training_mode) {
            const auto next_actions =
                result.done ? std::vector<core::Action>{}
                            : environment.get_valid_actions();
            agent.observe_transition(
                state,
                action,
                result.reward,
                result.next_state,
                result.done,
                next_actions);
        }

        state = result.next_state;
        if (result.done) {
            break;
        }
    }

    if (training_mode) {
        agent.end_episode();
    }

    const auto stats = core::EpisodeStats{
        episode_number,
        total_reward,
        steps,
        solved,
        agent.current_epsilon(),
        0.0,
        0.0,
    };

    if (record_trace) {
        trace_recorder->end_episode(stats);
    }

    return stats;
}

}  // namespace rl::training
