#include "render/dashboard.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "utils/filesystem.hpp"

namespace rl::render {

namespace {

#ifndef CPP_RL_LAB_SOURCE_DIR
#define CPP_RL_LAB_SOURCE_DIR "."
#endif

std::string escape_json(const std::string& value) {
    std::ostringstream stream;
    for (const char character : value) {
        switch (character) {
            case '\\':
                stream << "\\\\";
                break;
            case '"':
                stream << "\\\"";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                stream << character;
                break;
        }
    }
    return stream.str();
}

void write_string(std::ostream& output, const std::string& value) {
    output << '"' << escape_json(value) << '"';
}

void write_bool(std::ostream& output, const bool value) {
    output << (value ? "true" : "false");
}

void write_episode_metrics(
    std::ostream& output,
    const std::vector<core::EpisodeStats>& metrics) {
    output << '[';
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index > 0) {
            output << ',';
        }

        const auto& item = metrics[index];
        output << '{';
        output << "\"episode\":" << item.episode << ',';
        output << "\"totalReward\":" << item.total_reward << ',';
        output << "\"steps\":" << item.steps << ',';
        output << "\"solved\":";
        write_bool(output, item.solved);
        output << ',';
        output << "\"epsilon\":" << item.epsilon << ',';
        output << "\"movingAverageReward\":" << item.moving_average_reward << ',';
        output << "\"successRate\":" << item.success_rate;
        output << '}';
    }
    output << ']';
}

void write_fields(
    std::ostream& output,
    const std::vector<DashboardField>& fields) {
    output << '[';
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index > 0) {
            output << ',';
        }

        output << '{';
        output << "\"label\":";
        write_string(output, fields[index].label);
        output << ',';
        output << "\"value\":";
        write_string(output, fields[index].value);
        output << '}';
    }
    output << ']';
}

void write_rows(std::ostream& output, const std::vector<std::string>& rows) {
    output << '[';
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (index > 0) {
            output << ',';
        }
        write_string(output, rows[index]);
    }
    output << ']';
}

void write_evaluation_points(
    std::ostream& output,
    const std::vector<EvaluationPoint>& points) {
    output << '[';
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (index > 0) {
            output << ',';
        }
        const auto& point = points[index];
        output << '{';
        output << "\"episode\":" << point.episode << ',';
        output << "\"averageReward\":" << point.average_reward << ',';
        output << "\"successRate\":" << point.success_rate << ',';
        output << "\"averageSteps\":" << point.average_steps;
        output << '}';
    }
    output << ']';
}

void write_state_action_values(
    std::ostream& output,
    const std::vector<std::vector<double>>& values) {
    output << '[';
    for (std::size_t state = 0; state < values.size(); ++state) {
        if (state > 0) {
            output << ',';
        }

        output << '[';
        for (std::size_t action = 0; action < values[state].size(); ++action) {
            if (action > 0) {
                output << ',';
            }
            output << values[state][action];
        }
        output << ']';
    }
    output << ']';
}

void write_traces(std::ostream& output, const std::vector<EpisodeTrace>& traces) {
    output << '[';
    for (std::size_t trace_index = 0; trace_index < traces.size(); ++trace_index) {
        if (trace_index > 0) {
            output << ',';
        }

        const auto& trace = traces[trace_index];
        output << '{';
        output << "\"episode\":" << trace.episode << ',';
        output << "\"phase\":";
        write_string(output, trace.phase);
        output << ',';
        output << "\"trainingMode\":";
        write_bool(output, trace.training_mode);
        output << ',';
        output << "\"initialState\":" << trace.initial_state << ',';
        output << "\"totalReward\":" << trace.total_reward << ',';
        output << "\"steps\":" << trace.steps << ',';
        output << "\"solved\":";
        write_bool(output, trace.solved);
        output << ',';
        output << "\"epsilon\":" << trace.epsilon << ',';
        output << "\"stepsTrace\":[";

        for (std::size_t step_index = 0; step_index < trace.steps_trace.size();
             ++step_index) {
            if (step_index > 0) {
                output << ',';
            }

            const auto& step = trace.steps_trace[step_index];
            output << '{';
            output << "\"index\":" << step.index << ',';
            output << "\"state\":" << step.state << ',';
            output << "\"nextState\":" << step.next_state << ',';
            output << "\"action\":";
            write_string(output, step.action);
            output << ',';
            output << "\"reward\":" << step.reward << ',';
            output << "\"cumulativeReward\":" << step.cumulative_reward << ',';
            output << "\"done\":";
            write_bool(output, step.done);
            output << ',';
            output << "\"solved\":";
            write_bool(output, step.solved);
            output << ',';
            output << "\"truncated\":";
            write_bool(output, step.truncated);
            output << ',';
            output << "\"blocked\":";
            write_bool(output, step.blocked);
            output << ',';
            output << "\"validActions\":[";
            for (std::size_t action_index = 0;
                 action_index < step.valid_actions.size();
                 ++action_index) {
                if (action_index > 0) {
                    output << ',';
                }
                write_string(output, step.valid_actions[action_index]);
            }
            output << "],";
            output << "\"actionValues\":[";
            for (std::size_t value_index = 0;
                 value_index < step.action_values.size();
                 ++value_index) {
                if (value_index > 0) {
                    output << ',';
                }
                output << step.action_values[value_index];
            }
            output << ']';
            output << '}';
        }

        output << ']';
        output << '}';
    }
    output << ']';
}

void copy_asset_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
    if (!std::filesystem::exists(source)) {
        throw std::runtime_error("dashboard asset is missing: " + source.string());
    }

    std::filesystem::copy_file(
        source,
        destination,
        std::filesystem::copy_options::overwrite_existing);
}

std::filesystem::path assets_root() {
    return std::filesystem::path(CPP_RL_LAB_SOURCE_DIR) / "assets" /
           "visualizer";
}

}  // namespace

std::filesystem::path export_dashboard_bundle(
    const std::filesystem::path& output_dir,
    const DashboardData& data) {
    utils::ensure_directory(output_dir);

    const auto root = assets_root();
    const auto dashboard_path = output_dir / "dashboard.html";
    const auto stylesheet_path = output_dir / "dashboard.css";
    const auto script_path = output_dir / "dashboard.js";
    const auto data_path = output_dir / "dashboard_data.js";

    copy_asset_file(root / "dashboard.html", dashboard_path);
    copy_asset_file(root / "dashboard.css", stylesheet_path);
    copy_asset_file(root / "dashboard.js", script_path);

    std::ofstream output(data_path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to write dashboard data to " + data_path.string());
    }

    output << std::fixed << std::setprecision(8);
    output << "window.CPP_RL_LAB_DATA = {\n";
    output << "  \"title\":";
    write_string(output, data.title);
    output << ",\n  \"subtitle\":";
    write_string(output, data.subtitle);
    output << ",\n  \"mode\":";
    write_string(output, data.mode);
    output << ",\n  \"agentName\":";
    write_string(output, data.agent_name);
    output << ",\n  \"mazeSource\":";
    write_string(output, data.maze_source);
    output << ",\n  \"policyPath\":";
    write_string(output, data.policy_path);
    output << ",\n  \"bestPolicyPath\":";
    write_string(output, data.best_policy_path);
    output << ",\n  \"metricsPath\":";
    write_string(output, data.metrics_path);
    output << ",\n  \"seed\":" << data.seed;
    output << ",\n  \"maxSteps\":" << data.max_steps;
    output << ",\n  \"rewards\":{";
    output << "\"goalReward\":" << data.rewards.goal_reward << ',';
    output << "\"stepPenalty\":" << data.rewards.step_penalty << ',';
    output << "\"invalidMovePenalty\":" << data.rewards.invalid_move_penalty;
    output << '}';
    output << ",\n  \"layout\":{";
    output << "\"width\":" << data.layout.width << ',';
    output << "\"height\":" << data.layout.height << ',';
    output << "\"rows\":";
    write_rows(output, data.layout.rows);
    output << '}';
    output << ",\n  \"summary\":";
    write_fields(output, data.summary_fields);
    output << ",\n  \"configuration\":";
    write_fields(output, data.configuration_fields);
    output << ",\n  \"metrics\":";
    write_episode_metrics(output, data.episode_metrics);
    output << ",\n  \"evaluationPoints\":";
    write_evaluation_points(output, data.evaluation_points);
    output << ",\n  \"traces\":";
    write_traces(output, data.traces);
    output << ",\n  \"stateActionValues\":";
    write_state_action_values(output, data.state_action_values);
    output << "\n};\n";

    return dashboard_path;
}

}  // namespace rl::render
