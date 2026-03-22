# cpp-rl-lab

`cpp-rl-lab` is a modern C++ reinforcement learning lab built around a deterministic Maze Solver environment. The V1 goal is to provide a clean, reproducible foundation for experimenting with RL algorithms, not just a one-off maze demo.

It currently ships with:

- a reusable `Environment` / `Agent` / `Trainer` architecture
- a text-defined grid maze environment
- a tabular Q-learning agent
- a random baseline agent
- training and evaluation CLI workflows
- CSV metrics export and policy checkpointing
- an interactive visual dashboard generated for every run
- sample mazes, tests, and architecture docs

## Architecture

Top-level modules:

- `include/core`, `src/core`: shared RL types and action definitions
- `include/env`, `src/env`: environment interfaces, maze parsing, maze dynamics
- `include/agents`, `src/agents`: agent abstractions, random baseline, Q-learning
- `include/training`, `src/training`: episode loops, evaluation, checkpointing
- `include/metrics`, `src/metrics`: CSV metrics logging
- `include/render`, `src/render`: episode trace capture and interactive dashboard export
- `include/cli`, `src/cli`: command parsing
- `tests`: parser, environment, agent, and trainer behavior tests
- `configs/mazes`: sample human-readable maze layouts
- `docs/architecture.md`: design rationale and extension points

For a deeper design breakdown, see [docs/architecture.md](docs/architecture.md).

## Build

Standard CMake build:

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

If `cmake` is not on your `PATH`, the repo includes a helper script that also checks a common CLion-bundled CMake location:

```bash
./scripts/build.sh
```

## Test

Run the full test suite after building:

```bash
ctest --test-dir build --output-on-failure
```

Or use the helper:

```bash
./scripts/test.sh
```

The test suite covers:

- maze parsing and validation
- reset and transition behavior
- wall collision handling
- terminal conditions
- Q-learning Bellman update correctness
- deterministic seeded behavior
- a small end-to-end learning-vs-random regression
- dashboard bundle export and trace capture smoke coverage

## CLI Usage

### Train

```bash
./build/cpp_rl_lab train --maze configs/mazes/basic.txt --episodes 2000 --seed 42 --output-dir outputs/basic_train
```

### Evaluate a Saved Policy

```bash
./build/cpp_rl_lab eval --maze configs/mazes/basic.txt --policy outputs/basic_train/final_policy.qtable --episodes 100 --seed 42
```

### Run the Random Baseline

```bash
./build/cpp_rl_lab random --maze configs/mazes/basic.txt --episodes 100 --seed 42 --output-dir outputs/basic_random
```

### Helpful Training Flags

```bash
./build/cpp_rl_lab train \
  --maze configs/mazes/complex.txt \
  --episodes 3000 \
  --max-steps 200 \
  --eval-interval 100 \
  --eval-episodes 50 \
  --learning-rate 0.1 \
  --discount 0.95 \
  --epsilon-start 1.0 \
  --epsilon-min 0.05 \
  --epsilon-decay 0.995 \
  --trace-interval 100 \
  --dashboard-episodes 5 \
  --seed 42 \
  --output-dir outputs/complex_train
```

### Open the Interactive Dashboard

Every `train`, `eval`, and `random` run now writes a static visual dashboard in its output directory.

After training:

```bash
./build/cpp_rl_lab train --maze configs/mazes/basic.txt --episodes 1500 --seed 42 --output-dir outputs/basic_train
xdg-open outputs/basic_train/dashboard.html
```

If you prefer serving it over a local HTTP server:

```bash
./scripts/serve_dashboard.sh outputs/basic_train
```

The dashboard shows:

- episode reward and moving-average charts
- success rate and epsilon curves
- evaluation checkpoints
- an interactive maze playback panel with play, pause, step, and scrub controls
- traced training/evaluation/random episodes
- per-step decisions, rewards, valid actions, wall collisions, and terminal signals
- state-action value bars and a maze heatmap when the active agent exposes Q-values

## Maze Format

Mazes are plain text grids using:

- `#` for walls
- `S` for the single start cell
- `G` for the single goal cell
- `.` for free cells

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
- exactly one `S` and one `G` must exist
- transitions are deterministic
- the episode ends on reaching the goal or hitting the max step limit

If no `--maze` file is provided, the program uses a built-in default maze that matches the included basic sample.

## Q-Learning in V1

The V1 learner is a tabular Q-learning agent with:

- epsilon-greedy exploration
- configurable learning rate
- configurable discount factor
- configurable epsilon start / minimum / decay
- plain-text checkpoint save/load

State representation:

- the state is the flattened `(row, col)` position of the agent in the maze grid

Action space:

- `up`, `down`, `left`, `right`

Reward function:

- `+10.0` on goal
- `-0.1` per step
- `-0.75` extra penalty for invalid moves

Update rule:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

For terminal transitions, the bootstrapped future term is omitted.

## Outputs

Training writes:

- `training_metrics.csv`
- `final_policy.qtable`
- `best_policy.qtable`
- `dashboard.html`
- `dashboard.css`
- `dashboard.js`
- `dashboard_data.js`

Evaluation and random runs write their own CSV files inside the chosen output directory.

Each CSV row includes at least:

- `episode`
- `total_reward`
- `steps`
- `solved`
- `epsilon`

The logger also writes moving-average reward and running success rate to make experiment trends easier to inspect.

The dashboard data file embeds:

- run summary cards
- configuration details
- full episode metrics
- evaluation checkpoints
- traced episodes for playback
- state-action values when available

## Included Mazes

- `configs/mazes/basic.txt`: small starter maze
- `configs/mazes/complex.txt`: larger maze with a longer path and more walls

## Current Limitations

- only tabular agents are implemented in V1
- the environment is deterministic and single-agent
- the dashboard is a static exported web app rather than a live in-process renderer
- very long runs can generate large dashboard trace files if you choose aggressive trace settings
- checkpoints use a simple custom text format rather than a richer experiment manifest
- CLI configuration is flag-based rather than config-file driven

## Roadmap Ideas for V2

- SARSA and Monte Carlo control agents
- DQN with replay and target networks
- multiple environments beyond mazes
- richer experiment configuration files
- run summaries, plots, and policy trajectory visualization
- live graphical rendering or a desktop/web control panel
