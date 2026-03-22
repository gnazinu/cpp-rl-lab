# cpp-rl-lab

`cpp-rl-lab` is a reinforcement learning project in modern C++ built to teach, not just to demo.

The core idea is simple: train an agent to solve grid mazes, then inspect exactly how and why it improves. Under the hood, the repo is structured like a small RL framework, with clean environment and agent abstractions, reproducible experiment outputs, tests, and an interactive visual dashboard for replaying what the agent does.

It is meant to be useful for:

- learning reinforcement learning through a concrete, understandable environment
- learning how to structure RL code in clean modern C++
- exploring Q-learning, exploration, reward shaping, and evaluation
- building a portfolio-ready systems + AI project that is easy to extend

## What You Learn From This Project

This project is designed so that running it teaches you real RL ideas, not just maze mechanics.

By working through `cpp-rl-lab`, you can learn:

- how an RL problem is modeled as `state`, `action`, `reward`, and `termination`
- how to design an `Environment` interface that is reusable across tasks
- how epsilon-greedy exploration changes agent behavior over time
- how tabular Q-learning updates a policy from experience
- how reward design affects learning speed and final behavior
- how to compare a learned policy against a random baseline
- how to log reproducible metrics and checkpoints for experiments
- how to inspect agent decisions visually instead of treating training like a black box

In other words, this repo is both:

- a working maze-solving RL system
- a compact reference implementation of RL architecture in C++

## What The Project Does

V1 includes:

- a reusable `Environment` / `Agent` / `Trainer` architecture
- a deterministic grid-based maze environment
- a tabular Q-learning agent
- a random baseline agent
- training, evaluation, and baseline CLI workflows
- CSV metrics export
- checkpoint saving/loading for the learned Q-table
- sample maze files
- tests for environment, agent, training, and render-related behavior
- an interactive dashboard generated for every run

This means you can:

1. train an agent on a maze
2. save the learned policy
3. evaluate that policy
4. compare it against a random agent
5. open a dashboard to replay episodes step by step
6. inspect rewards, success rate, epsilon decay, and value estimates

## Why The Visual Dashboard Matters

A lot of RL projects tell you only whether the reward went up.

This one also lets you see:

- where the agent moves in the maze
- when it hits walls or wastes steps
- how training episodes differ from greedy evaluation episodes
- how Q-values change the decisions it prefers
- how exploration decays over time
- how learned behavior compares with random behavior

Each run exports a static dashboard to the output directory, so you can open it locally without adding GUI dependencies to the C++ project.

## Quick Start

Build:

```bash
./scripts/build.sh
```

Run tests:

```bash
./scripts/test.sh
```

Train on the basic maze:

```bash
./build/cpp_rl_lab train \
  --maze configs/mazes/basic.txt \
  --episodes 400 \
  --seed 42 \
  --trace-interval 80 \
  --dashboard-episodes 4 \
  --output-dir outputs/basic_train
```

Open the dashboard:

```bash
xdg-open outputs/basic_train/dashboard.html
```

Evaluate the saved policy:

```bash
./build/cpp_rl_lab eval \
  --maze configs/mazes/basic.txt \
  --policy outputs/basic_train/final_policy.qtable \
  --episodes 40 \
  --seed 42 \
  --dashboard-episodes 4 \
  --output-dir outputs/basic_eval
```

Run the random baseline:

```bash
./build/cpp_rl_lab random \
  --maze configs/mazes/basic.txt \
  --episodes 40 \
  --seed 42 \
  --dashboard-episodes 4 \
  --output-dir outputs/basic_random
```

Serve dashboards over a local HTTP server if you prefer:

```bash
./scripts/serve_dashboard.sh outputs/basic_train
```

## CLI Overview

Main commands:

- `train`: learn a Q-table from repeated episodes
- `eval`: load a saved policy and run greedy evaluation
- `random`: run the baseline random agent for comparison

Examples:

```bash
./build/cpp_rl_lab train --maze configs/mazes/basic.txt --episodes 2000 --seed 42 --output-dir outputs/train_run
./build/cpp_rl_lab eval --maze configs/mazes/basic.txt --policy outputs/train_run/final_policy.qtable --episodes 100 --seed 42 --output-dir outputs/eval_run
./build/cpp_rl_lab random --maze configs/mazes/basic.txt --episodes 100 --seed 42 --output-dir outputs/random_run
```

Useful training flags:

```bash
--max-steps <n>
--eval-interval <n>
--eval-episodes <n>
--learning-rate <value>
--discount <value>
--epsilon-start <value>
--epsilon-min <value>
--epsilon-decay <value>
--trace-interval <n>
--dashboard-episodes <n>
```

## How The RL Part Works

### Environment

The environment is a deterministic maze/gridworld.

- state: the agent position, represented as a flattened cell index
- actions: `up`, `down`, `left`, `right`
- transition: movement to adjacent cells unless blocked by a wall
- terminal conditions:
  - reaching the goal
  - exceeding the max step limit

### Rewards

The default reward design is:

- `+10.0` for reaching the goal
- `-0.1` per step
- `-0.75` additional penalty for invalid moves

This encourages the agent to:

- solve the maze
- solve it efficiently
- avoid repeatedly colliding with walls

### Q-Learning

The learning agent uses tabular Q-learning with epsilon-greedy exploration.

Update rule:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

Configurable parameters:

- learning rate
- discount factor
- epsilon start
- epsilon minimum
- epsilon decay

The learned Q-table is saved in a plain-text format so it is easy to inspect and version.

## Interactive Dashboard

Every run exports:

- `dashboard.html`
- `dashboard.css`
- `dashboard.js`
- `dashboard_data.js`

The dashboard includes:

- reward per episode chart
- moving average reward chart
- success rate and epsilon curves
- evaluation checkpoints
- episode trace library
- interactive playback with play, pause, step, and slider controls
- current-step decision details
- wall-collision and terminal-state visibility
- Q-value bars and value heatmap when available

This makes the project much easier to understand because you can watch the agent act, not just read scalar metrics.

## Maze Format

Maze files are plain text and human-readable:

- `#` = wall
- `S` = start
- `G` = goal
- `.` = free cell

Example:

```text
########
#S.....#
#.###..#
#...#G.#
########
```

Rules:

- all rows must have the same width
- there must be exactly one `S`
- there must be exactly one `G`
- transitions are deterministic

Included samples:

- `configs/mazes/basic.txt`
- `configs/mazes/complex.txt`

## Project Structure

Main modules:

- `include/core`, `src/core`: shared RL types and action definitions
- `include/env`, `src/env`: environment abstractions, maze parsing, maze dynamics
- `include/agents`, `src/agents`: baseline and learning agents
- `include/training`, `src/training`: training and evaluation loops
- `include/metrics`, `src/metrics`: CSV metrics logging
- `include/render`, `src/render`: trace capture and dashboard export
- `include/cli`, `src/cli`: command-line parsing and workflow entry points
- `tests`: unit and integration-style tests
- `configs/mazes`: sample mazes
- `docs/architecture.md`: design notes and extension points

For a deeper architecture walkthrough, see [docs/architecture.md](docs/architecture.md).

## Build And Test

Standard CMake:

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Helper scripts:

```bash
./scripts/build.sh
./scripts/test.sh
```

The test suite covers:

- maze parsing and validation
- reset and transition correctness
- wall collisions
- terminal conditions
- Q-learning update correctness
- deterministic seeded behavior
- learning-vs-random regression
- dashboard export and trace capture smoke coverage

## Outputs

Training runs generate artifacts such as:

- `training_metrics.csv`
- `final_policy.qtable`
- `best_policy.qtable`
- dashboard files

Evaluation and random runs generate their own metrics CSVs and dashboards in their output directories.

Metrics include at least:

- `episode`
- `total_reward`
- `steps`
- `solved`
- `epsilon`
- moving average reward
- running success rate

## Who This Project Is For

This project is a good fit if you want:

- a first serious RL codebase in C++
- a clean example of tabular RL architecture
- a portfolio project that mixes systems engineering and AI
- a practical way to understand exploration, rewards, and policy improvement
- a base for adding SARSA, DQN, new environments, or richer rendering later

## Current Limitations

V1 is intentionally focused and lightweight.

Current limitations:

- only tabular agents are implemented
- the environment is deterministic and single-agent
- the dashboard is exported as a static web bundle rather than a live embedded UI
- very aggressive tracing can produce larger dashboard data files
- configuration is CLI-driven rather than config-file driven
- checkpoints use a simple custom text format

## Roadmap Ideas

Good V2 directions include:

- SARSA and Monte Carlo control
- DQN and replay-buffer-based agents
- more environments beyond mazes
- experiment config files
- side-by-side policy comparison dashboards
- richer rendering and live control panels
- multi-run experiment summaries and plots
