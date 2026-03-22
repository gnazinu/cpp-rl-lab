#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/action.hpp"
#include "core/types.hpp"

namespace rl::env {

class Environment {
  public:
    virtual ~Environment() = default;

    virtual int reset() = 0;
    virtual core::StepResult step(core::Action action) = 0;
    virtual bool is_terminal() const = 0;
    virtual int get_state() const = 0;
    virtual std::vector<core::Action> get_action_space() const = 0;
    virtual std::vector<core::Action> get_valid_actions() const = 0;
    virtual std::size_t state_space_size() const = 0;
    virtual std::size_t action_space_size() const = 0;
    virtual void seed(std::uint64_t seed) = 0;
    virtual std::string render() const = 0;
};

}  // namespace rl::env
