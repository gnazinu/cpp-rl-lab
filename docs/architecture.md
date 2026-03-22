# Architecture Overview

## Goals

`cpp-rl-lab` is structured as a small but extensible reinforcement learning lab:

- deterministic environments for reproducible experiments
- clean agent and environment interfaces
- simple experiment orchestration and metrics output
- clear seams for future algorithms and environments

## Module Responsibilities

### `core`

- shared domain types such as `Action`, `Position`, `StepResult`, and `EpisodeStats`
- lightweight utilities that are independent of any specific environment or agent

### `env`

- `Environment` interface defining the common RL interaction loop
- `MazeLayout` for parsing and validating text mazes
- `MazeEnvironment` for deterministic grid-world transitions, rewards, rendering, and episode termination

### `agents`

- `Agent` interface for policy selection, transition observation, persistence, and seeding
- `RandomAgent` as the baseline policy
- `QLearningAgent` as the tabular control algorithm for V1

### `training`

- `Trainer` for training and evaluation loops
- periodic evaluation and checkpointing
- episode-level summaries and separation of train vs eval behavior

### `metrics`

- `MetricsLogger` for per-episode CSV export
- moving-average reward and running success-rate tracking

### `cli`

- command parsing and mode selection
- configuration handoff into the runtime modules

### `utils`

- small cross-cutting helpers such as directory creation

## Class Relationships

- `Trainer` depends on the abstract `Environment` and `Agent` interfaces rather than concrete implementations.
- `MazeEnvironment` implements `Environment`.
- `RandomAgent` and `QLearningAgent` implement `Agent`.
- `MetricsLogger` is injected into training or evaluation runs so metrics export stays independent from the environment and agent logic.
- `main.cpp` composes the concrete maze, agent, trainer, and logger for each CLI command.

## Design Decisions

### State Representation

V1 uses a discrete integer state equal to the flattened `(row, col)` maze position. This keeps the state space explicit and directly compatible with a tabular Q-table.

### Action Handling

The global action space is fixed to `up`, `down`, `left`, and `right`. The environment also exposes valid actions from the current state so agents can choose between "sample from all actions" and "sample from valid actions only" policies cleanly.

### Reward Scheme

The maze uses:

- `+10.0` goal reward
- `-0.1` per step
- `-0.75` additional penalty for invalid moves

This gives the agent a sparse success signal while still nudging it toward shorter solutions and away from repeatedly hitting walls.

### Persistence Format

The Q-table is stored in a simple text format. That keeps checkpoints human-readable, diffable, and easy to debug without bringing in a JSON dependency.

### Testing Strategy

The V1 test setup uses a lightweight internal test executable wired into CTest. That keeps the build dependency-free while still providing meaningful coverage of parsing, environment transitions, learning updates, and training behavior.

## Extension Points

The current structure is meant to make these follow-on upgrades straightforward:

- SARSA: add a new tabular agent that reuses the same environment, metrics, and CLI wiring
- DQN: introduce a new agent implementation and a replay buffer module without changing the environment interface
- new environments: implement `Environment` for additional tasks such as cliff-walking, cart-pole approximations, or custom games
- rendering layer: add dedicated ASCII or graphical renderers under `render/`
- experiment configs: load hyperparameters and run settings from config files rather than only CLI flags

## V2 Ideas

- richer checkpoint metadata and experiment manifests
- batched experiment sweeps
- stochastic transitions and reward variants
- policy visualization and trajectory playback
- richer evaluation reports and comparative baselines
