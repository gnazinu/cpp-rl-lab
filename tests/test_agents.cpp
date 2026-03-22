#include "agents/q_learning_agent.hpp"
#include "agents/random_agent.hpp"
#include "core/action.hpp"
#include "test_framework.hpp"

RL_TEST_CASE("q-learning update follows the Bellman backup") {
    rl::agents::QLearningConfig config;
    config.state_space_size = 5;
    config.learning_rate = 0.5;
    config.discount_factor = 0.9;
    config.epsilon_start = 0.0;
    config.epsilon_min = 0.0;
    config.epsilon_decay = 1.0;

    rl::agents::QLearningAgent agent(config);
    agent.set_q_value(2, rl::core::Action::Up, 4.0);

    agent.observe_transition(
        1,
        rl::core::Action::Right,
        1.0,
        2,
        false,
        {rl::core::Action::Up, rl::core::Action::Left});

    RL_EXPECT_NEAR(agent.q_value(1, rl::core::Action::Right), 2.3, 1e-9);
}

RL_TEST_CASE("random agent is deterministic when seeded identically") {
    rl::agents::RandomAgent first_agent;
    rl::agents::RandomAgent second_agent;
    first_agent.seed(1234);
    second_agent.seed(1234);

    const std::vector<rl::core::Action> available_actions{
        rl::core::Action::Up,
        rl::core::Action::Down,
        rl::core::Action::Left,
    };

    for (int index = 0; index < 8; ++index) {
        const auto first =
            static_cast<int>(first_agent.select_action(0, available_actions, true));
        const auto second = static_cast<int>(
            second_agent.select_action(0, available_actions, true));
        RL_EXPECT_EQ(first, second);
    }
}
