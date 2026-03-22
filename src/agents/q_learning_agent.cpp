#include "agents/q_learning_agent.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "core/action.hpp"

namespace rl::agents {

QLearningAgent::QLearningAgent(QLearningConfig config)
    : config_(std::move(config)),
      q_table_(
          config_.state_space_size,
          std::vector<double>(core::action_count(), 0.0)),
      epsilon_(config_.epsilon_start),
      rng_(std::random_device{}()) {
    if (config_.learning_rate <= 0.0 || config_.learning_rate > 1.0) {
        throw std::runtime_error("learning_rate must be in (0, 1]");
    }
    if (config_.discount_factor < 0.0 || config_.discount_factor > 1.0) {
        throw std::runtime_error("discount_factor must be in [0, 1]");
    }
    if (config_.epsilon_start < 0.0 || config_.epsilon_min < 0.0) {
        throw std::runtime_error("epsilon values must be non-negative");
    }
    if (config_.epsilon_decay <= 0.0 || config_.epsilon_decay > 1.0) {
        throw std::runtime_error("epsilon_decay must be in (0, 1]");
    }
    if (config_.epsilon_min > config_.epsilon_start) {
        throw std::runtime_error(
            "epsilon_min must be less than or equal to epsilon_start");
    }
}

std::string QLearningAgent::name() const {
    return "q_learning";
}

void QLearningAgent::seed(const std::uint64_t seed) {
    rng_.seed(static_cast<std::mt19937::result_type>(seed));
}

core::Action QLearningAgent::select_action(
    const int state,
    const std::vector<core::Action>& available_actions,
    const bool training_mode) {
    validate_state(state);

    const auto candidates = normalized_actions(available_actions);

    if (training_mode) {
        std::uniform_real_distribution<double> probability(0.0, 1.0);
        if (probability(rng_) < epsilon_) {
            std::uniform_int_distribution<std::size_t> random_index(
                0,
                candidates.size() - 1);
            return candidates[random_index(rng_)];
        }
    }

    double best_value = -std::numeric_limits<double>::infinity();
    std::vector<core::Action> best_actions;

    for (const auto action : candidates) {
        const double value = q_table_[static_cast<std::size_t>(state)]
                                    [core::to_index(action)];
        if (value > best_value) {
            best_value = value;
            best_actions = {action};
        } else if (value == best_value) {
            best_actions.push_back(action);
        }
    }

    std::uniform_int_distribution<std::size_t> best_index(
        0,
        best_actions.size() - 1);
    return best_actions[best_index(rng_)];
}

void QLearningAgent::observe_transition(
    const int state,
    const core::Action action,
    const double reward,
    const int next_state,
    const bool done,
    const std::vector<core::Action>& next_available_actions) {
    validate_state(state);
    validate_state(next_state);

    double next_max = 0.0;
    if (!done) {
        const auto candidates = normalized_actions(next_available_actions);
        next_max = -std::numeric_limits<double>::infinity();
        for (const auto next_action : candidates) {
            next_max = std::max(
                next_max,
                q_table_[static_cast<std::size_t>(next_state)]
                        [core::to_index(next_action)]);
        }
    }

    double& current =
        q_table_[static_cast<std::size_t>(state)][core::to_index(action)];
    const double target =
        reward + (done ? 0.0 : config_.discount_factor * next_max);
    current += config_.learning_rate * (target - current);
}

void QLearningAgent::end_episode() {
    epsilon_ = std::max(config_.epsilon_min, epsilon_ * config_.epsilon_decay);
}

void QLearningAgent::save(const std::filesystem::path& path) const {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to save q-table to " + path.string());
    }

    output << "qtable_v1\n";
    output << "states " << q_table_.size() << '\n';
    output << "actions " << core::action_count() << '\n';
    output << "learning_rate " << config_.learning_rate << '\n';
    output << "discount_factor " << config_.discount_factor << '\n';
    output << "epsilon " << epsilon_ << '\n';
    output << "epsilon_min " << config_.epsilon_min << '\n';
    output << "epsilon_decay " << config_.epsilon_decay << '\n';

    for (std::size_t state = 0; state < q_table_.size(); ++state) {
        output << "row " << state;
        for (const double value : q_table_[state]) {
            output << ' ' << value;
        }
        output << '\n';
    }
}

void QLearningAgent::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to load q-table from " + path.string());
    }

    std::string header;
    input >> header;
    if (header != "qtable_v1") {
        throw std::runtime_error("unsupported q-table file format");
    }

    std::size_t states = 0;
    std::size_t actions = 0;
    std::string label;
    input >> label >> states;
    if (label != "states") {
        throw std::runtime_error("malformed q-table: missing states header");
    }
    input >> label >> actions;
    if (label != "actions") {
        throw std::runtime_error("malformed q-table: missing actions header");
    }
    if (actions != core::action_count()) {
        throw std::runtime_error("q-table action count does not match build");
    }

    if (q_table_.empty()) {
        config_.state_space_size = states;
        q_table_.assign(states, std::vector<double>(actions, 0.0));
    } else if (states != q_table_.size()) {
        throw std::runtime_error(
            "q-table state count does not match environment");
    }

    input >> label >> config_.learning_rate;
    if (label != "learning_rate") {
        throw std::runtime_error("malformed q-table: missing learning_rate");
    }
    input >> label >> config_.discount_factor;
    if (label != "discount_factor") {
        throw std::runtime_error("malformed q-table: missing discount_factor");
    }
    input >> label >> epsilon_;
    if (label != "epsilon") {
        throw std::runtime_error("malformed q-table: missing epsilon");
    }
    input >> label >> config_.epsilon_min;
    if (label != "epsilon_min") {
        throw std::runtime_error("malformed q-table: missing epsilon_min");
    }
    input >> label >> config_.epsilon_decay;
    if (label != "epsilon_decay") {
        throw std::runtime_error("malformed q-table: missing epsilon_decay");
    }

    for (std::size_t expected_state = 0; expected_state < states; ++expected_state) {
        std::string row_label;
        std::size_t state_index = 0;
        input >> row_label >> state_index;
        if (row_label != "row" || state_index != expected_state) {
            throw std::runtime_error("malformed q-table row ordering");
        }

        for (std::size_t action_index = 0; action_index < actions; ++action_index) {
            input >> q_table_[state_index][action_index];
        }
    }
}

double QLearningAgent::current_epsilon() const {
    return epsilon_;
}

double QLearningAgent::q_value(const int state, const core::Action action) const {
    validate_state(state);
    return q_table_[static_cast<std::size_t>(state)][core::to_index(action)];
}

void QLearningAgent::set_q_value(
    const int state,
    const core::Action action,
    const double value) {
    validate_state(state);
    q_table_[static_cast<std::size_t>(state)][core::to_index(action)] = value;
}

std::size_t QLearningAgent::state_space_size() const {
    return q_table_.size();
}

void QLearningAgent::validate_state(const int state) const {
    if (state < 0 || static_cast<std::size_t>(state) >= q_table_.size()) {
        throw std::runtime_error("state index is out of q-table bounds");
    }
}

std::vector<core::Action> QLearningAgent::normalized_actions(
    const std::vector<core::Action>& available_actions) const {
    if (!available_actions.empty()) {
        return available_actions;
    }

    const auto& actions = core::all_actions();
    return {actions.begin(), actions.end()};
}

}  // namespace rl::agents
