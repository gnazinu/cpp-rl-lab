#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "env/maze_environment.hpp"
#include "env/maze_layout.hpp"
#include "render/trace_recorder.hpp"

namespace rl::render {

struct DashboardField {
    std::string label;
    std::string value;
};

struct EvaluationPoint {
    std::size_t episode = 0;
    double average_reward = 0.0;
    double success_rate = 0.0;
    double average_steps = 0.0;
};

struct DashboardData {
    std::string title;
    std::string subtitle;
    std::string mode;
    std::string agent_name;
    std::string maze_source;
    std::string policy_path;
    std::string best_policy_path;
    std::string metrics_path;
    std::uint64_t seed = 0;
    int max_steps = 0;
    env::MazeRewards rewards;
    env::MazeLayout layout;
    std::vector<DashboardField> summary_fields;
    std::vector<DashboardField> configuration_fields;
    std::vector<core::EpisodeStats> episode_metrics;
    std::vector<EvaluationPoint> evaluation_points;
    std::vector<EpisodeTrace> traces;
    std::vector<std::vector<double>> state_action_values;
};

std::filesystem::path export_dashboard_bundle(
    const std::filesystem::path& output_dir,
    const DashboardData& data);

}  // namespace rl::render
