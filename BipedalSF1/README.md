# BipedalSF1

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered task is:
- `Bipedalsf1-TRON2A-Balance-Walk-v0` — balance and velocity-commanded walking across flat, stair, and uneven terrain.


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

The default backend is Newton MJWarp. You can run on CPU or select a CUDA device with `--device`.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 16 --device cpu
uv run isaaclab random_agent --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 16 --device cuda:0

# Train with a live Newton MJWarp viewport (Newton GL)
uv run isaaclab train --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 64 --device cuda:0 --viz newton_gl physics=newton_mjwarp

# Train with the Isaac Sim / PhysX viewport (install the Isaac Sim extra)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 64 --device cuda:0 --viz kit physics=isaacsim_physx

# Play a trained checkpoint with either live viewport
uv run isaaclab play --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --checkpoint latest --device cuda:0 --viz newton_gl physics=newton_mjwarp
uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --checkpoint latest --device cuda:0 --viz kit physics=isaacsim_physx

# Small CPU training run
uv run isaaclab train --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 16 --device cpu

# Distributed training
uv run isaaclab train_multigpu --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task Bipedalsf1-TRON2A-Balance-Walk-v0 --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library rsl_rl --task Bipedalsf1-TRON2A-Balance-Walk-v0 --max_iterations 10
```

The Isaac Sim viewport uses `--viz kit`; the Newton MJWarp viewport uses `--viz newton_gl`. Visual rendering is off
unless `--viz` is passed. Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Task families live in `src/BipedalSF1/tasks`. The TRON2A manager-based environment, registration, and RSL-RL PPO
configuration live in `tasks/balance/config/tron2a`. Training environments are sampled from a mixed terrain set;
successful episodes move to harder terrain levels. A local height scanner gives the policy a compact terrain map, so
no depth camera is required for this terrain curriculum.

## Project assets

The layered SF-TRON2A USD lives in the repository-level `SF_TRON2A/usd` directory beside this package. Its layers and
configuration subdirectory must stay together. The asset module resolves `robot.usd` from an editable checkout:

```python
from BipedalSF1.assets import TRON2A_USD_PATH

robot_usd_path = TRON2A_USD_PATH
```

The environment uses the supplied USD directly, including its linked configuration layers and authored geometry.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
