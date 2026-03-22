#include "core/action.hpp"
#include "env/maze_environment.hpp"
#include "env/maze_layout.hpp"
#include "test_framework.hpp"

namespace {

rl::env::MazeLayout make_test_layout() {
    return rl::env::MazeLayout::from_lines({
        "#####",
        "#S.G#",
        "#...#",
        "#####",
    });
}

}  // namespace

RL_TEST_CASE("maze environment reset returns the start state") {
    rl::env::MazeEnvironment environment({make_test_layout(), 10, {}});

    const int state = environment.reset();

    RL_EXPECT_EQ(state, environment.layout().flatten(environment.layout().start));
    RL_EXPECT_TRUE(!environment.is_terminal());
    RL_EXPECT_EQ(environment.steps_taken(), 0);
}

RL_TEST_CASE("maze environment moves to adjacent walkable cells") {
    rl::env::MazeEnvironment environment({make_test_layout(), 10, {}});
    environment.reset();

    const auto result = environment.step(rl::core::Action::Right);

    RL_EXPECT_EQ(result.next_state, environment.layout().flatten({1, 2}));
    RL_EXPECT_NEAR(result.reward, -0.1, 1e-9);
    RL_EXPECT_TRUE(!result.done);
}

RL_TEST_CASE("maze environment penalizes wall collisions and keeps state") {
    rl::env::MazeEnvironment environment({make_test_layout(), 10, {}});
    const int start_state = environment.reset();

    const auto result = environment.step(rl::core::Action::Up);

    RL_EXPECT_EQ(result.next_state, start_state);
    RL_EXPECT_NEAR(result.reward, -0.85, 1e-9);
    RL_EXPECT_TRUE(!result.done);
}

RL_TEST_CASE("maze environment terminates on goal") {
    rl::env::MazeEnvironment environment({make_test_layout(), 10, {}});
    environment.reset();
    environment.step(rl::core::Action::Right);

    const auto result = environment.step(rl::core::Action::Right);

    RL_EXPECT_TRUE(result.done);
    RL_EXPECT_TRUE(result.solved);
    RL_EXPECT_NEAR(result.reward, 9.9, 1e-9);
}

RL_TEST_CASE("maze environment terminates when max steps are exhausted") {
    rl::env::MazeEnvironment environment({make_test_layout(), 1, {}});
    environment.reset();

    const auto result = environment.step(rl::core::Action::Right);

    RL_EXPECT_TRUE(result.done);
    RL_EXPECT_TRUE(result.truncated);
    RL_EXPECT_TRUE(!result.solved);
}
