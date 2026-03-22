#include <filesystem>
#include <fstream>
#include <string>

#include "core/action.hpp"
#include "render/dashboard.hpp"
#include "render/trace_recorder.hpp"
#include "test_framework.hpp"

RL_TEST_CASE("trace recorder captures episode transitions") {
    rl::render::TraceRecorder recorder({
        "training",
        1,
        10,
        true,
        true,
    });

    RL_EXPECT_TRUE(recorder.should_record_episode(1, 5));
    recorder.begin_episode(1, "training", true, 6, 1.0);
    recorder.record_step(
        1,
        6,
        rl::core::Action::Right,
        {
            7,
            -0.1,
            false,
            false,
            false,
        },
        {rl::core::Action::Down, rl::core::Action::Right},
        {0.1, 0.2, 0.3, 0.4});
    recorder.end_episode({
        1,
        -0.1,
        1,
        false,
        1.0,
        0.0,
        0.0,
    });

    RL_EXPECT_EQ(recorder.episodes().size(), static_cast<std::size_t>(1));
    RL_EXPECT_EQ(recorder.episodes().front().initial_state, 6);
    RL_EXPECT_EQ(
        recorder.episodes().front().steps_trace.front().action,
        std::string("right"));
    RL_EXPECT_NEAR(
        recorder.episodes().front().steps_trace.front().reward,
        -0.1,
        1e-9);
}

RL_TEST_CASE("dashboard exporter writes the static bundle") {
    const auto output_dir =
        std::filesystem::temp_directory_path() / "cpp_rl_lab_dashboard_test";

    rl::render::DashboardData data;
    data.title = "Test Dashboard";
    data.subtitle = "render smoke test";
    data.mode = "train";
    data.agent_name = "q_learning";
    data.maze_source = "built-in-default";
    data.metrics_path = "outputs/training_metrics.csv";
    data.seed = 42;
    data.max_steps = 100;
    data.layout = rl::env::MazeLayout::default_layout();
    data.summary_fields = {
        {"Avg Reward", "1.234"},
    };
    data.configuration_fields = {
        {"Mode", "train"},
    };
    data.episode_metrics = {
        {
            1,
            1.0,
            4,
            true,
            0.5,
            1.0,
            1.0,
        },
    };

    const auto dashboard_path =
        rl::render::export_dashboard_bundle(output_dir, data);

    RL_EXPECT_TRUE(std::filesystem::exists(dashboard_path));
    RL_EXPECT_TRUE(std::filesystem::exists(output_dir / "dashboard.css"));
    RL_EXPECT_TRUE(std::filesystem::exists(output_dir / "dashboard.js"));
    RL_EXPECT_TRUE(std::filesystem::exists(output_dir / "dashboard_data.js"));

    std::ifstream input(output_dir / "dashboard_data.js");
    std::string contents(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    RL_EXPECT_TRUE(contents.find("Test Dashboard") != std::string::npos);
}
