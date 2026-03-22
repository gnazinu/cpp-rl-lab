#include <filesystem>

#include "agents/q_learning_agent.hpp"
#include "agents/random_agent.hpp"
#include "env/maze_environment.hpp"
#include "env/maze_layout.hpp"
#include "metrics/metrics_logger.hpp"
#include "test_framework.hpp"
#include "training/trainer.hpp"

namespace {

rl::env::MazeLayout make_corridor_layout() {
    return rl::env::MazeLayout::from_lines({
        "#######",
        "#S...G#",
        "#######",
    });
}

}  // namespace

RL_TEST_CASE("trainer learns a simple maze better than the random baseline") {
    const auto layout = make_corridor_layout();

    rl::env::MazeEnvironment training_environment({layout, 10, {}});
    training_environment.seed(11);

    rl::agents::QLearningConfig q_config;
    q_config.state_space_size = training_environment.state_space_size();
    q_config.learning_rate = 0.2;
    q_config.discount_factor = 0.95;
    q_config.epsilon_start = 1.0;
    q_config.epsilon_min = 0.05;
    q_config.epsilon_decay = 0.98;

    rl::agents::QLearningAgent learning_agent(q_config);
    learning_agent.seed(12);

    rl::training::TrainerConfig trainer_config;
    trainer_config.episodes = 220;
    trainer_config.evaluation_interval = 55;
    trainer_config.evaluation_episodes = 20;
    trainer_config.output_dir =
        std::filesystem::temp_directory_path() / "cpp_rl_lab_training_test";

    rl::training::Trainer trainer(trainer_config);
    rl::metrics::MetricsLogger logger(
        trainer_config.output_dir / "training_metrics.csv",
        20);

    trainer.train(training_environment, learning_agent, logger);
    const auto learned_summary =
        trainer.evaluate(training_environment, learning_agent, nullptr, 50);

    rl::env::MazeEnvironment random_environment({layout, 10, {}});
    random_environment.seed(11);
    rl::agents::RandomAgent random_agent;
    random_agent.seed(12);
    const auto random_summary =
        trainer.evaluate(random_environment, random_agent, nullptr, 50);

    RL_EXPECT_TRUE(learned_summary.success_rate > random_summary.success_rate);
    RL_EXPECT_TRUE(learned_summary.success_rate >= 0.80);
}
