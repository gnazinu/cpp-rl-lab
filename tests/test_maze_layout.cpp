#include <vector>

#include "env/maze_layout.hpp"
#include "test_framework.hpp"

RL_TEST_CASE("maze layout parses rectangular maze and finds start/goal") {
    const auto layout = rl::env::MazeLayout::from_lines({
        "#####",
        "#S.G#",
        "#####",
    });

    RL_EXPECT_EQ(layout.width, 5);
    RL_EXPECT_EQ(layout.height, 3);
    RL_EXPECT_EQ(layout.start.row, 1);
    RL_EXPECT_EQ(layout.start.col, 1);
    RL_EXPECT_EQ(layout.goal.row, 1);
    RL_EXPECT_EQ(layout.goal.col, 3);
    RL_EXPECT_EQ(layout.flatten(layout.start), 6);
}

RL_TEST_CASE("maze layout validation rejects missing goal") {
    RL_EXPECT_THROW(rl::env::MazeLayout::from_lines({
        "#####",
        "#S..#",
        "#####",
    }));
}

RL_TEST_CASE("maze layout validation rejects jagged rows") {
    RL_EXPECT_THROW(rl::env::MazeLayout::from_lines({
        "#####",
        "#S.G#",
        "####",
    }));
}
