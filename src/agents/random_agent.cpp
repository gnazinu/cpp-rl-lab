#include "agents/random_agent.hpp"

#include <fstream>
#include <stdexcept>

#include "core/action.hpp"

namespace rl::agents {

RandomAgent::RandomAgent(RandomAgentConfig config)
    : config_(config), rng_(std::random_device{}()) {}

std::string RandomAgent::name() const {
    return "random";
}

void RandomAgent::seed(const std::uint64_t seed) {
    rng_.seed(static_cast<std::mt19937::result_type>(seed));
}

core::Action RandomAgent::select_action(
    int,
    const std::vector<core::Action>& available_actions,
    bool) {
    std::vector<core::Action> candidates;
    if (config_.valid_actions_only && !available_actions.empty()) {
        candidates = available_actions;
    } else {
        const auto& all_actions = core::all_actions();
        candidates.assign(all_actions.begin(), all_actions.end());
    }

    if (candidates.empty()) {
        throw std::runtime_error("random agent has no actions to sample");
    }

    std::uniform_int_distribution<std::size_t> distribution(
        0,
        candidates.size() - 1);
    return candidates[distribution(rng_)];
}

void RandomAgent::observe_transition(
    int,
    core::Action,
    double,
    int,
    bool,
    const std::vector<core::Action>&) {}

void RandomAgent::save(const std::filesystem::path& path) const {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(
            "failed to save random agent metadata to " + path.string());
    }

    output << "random_agent_v1\n";
    output << "valid_actions_only " << (config_.valid_actions_only ? 1 : 0)
           << '\n';
}

void RandomAgent::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "failed to load random agent metadata from " + path.string());
    }

    std::string header;
    input >> header;
    if (header != "random_agent_v1") {
        throw std::runtime_error("unsupported random agent file format");
    }

    std::string label;
    int valid_only = 1;
    input >> label >> valid_only;
    if (label != "valid_actions_only") {
        throw std::runtime_error("malformed random agent metadata");
    }
    config_.valid_actions_only = (valid_only != 0);
}

}  // namespace rl::agents
