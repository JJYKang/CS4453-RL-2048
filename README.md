# CS4453-RL-2048


## Environment
Core API and behavior reference for the 2048 Gymnasium environment.

### Module layout
- `src/rl2048/envs/game_2048_env.py`: Gymnasium `Env` implementation (`reset/step/render`)
- `src/rl2048/envs/logic.py`: board mechanics (move/merge/spawn/valid actions)
- `src/rl2048/envs/types.py`: shared env type definitions (`Obs`, `InfoDict`, `RewardFn`, `Board`)
- `src/rl2048/envs/__init__.py`: env exports and Gym registration helper
- `src/rl2048/rewards/merge_space_max_tile.py`: reusable reward function

### Registration and creation
```python
import gymnasium as gym
import rl2048  # registers Game2048-v0

env = gym.make("Game2048-v0", observation_mode="log2", reward_mode="score")
obs, info = env.reset(seed=0)
```

You can also instantiate directly:

```python
from rl2048.envs import Game2048Env

env = Game2048Env()
```

### Action space
- Discrete(4)
- Mapping:
  - `0`: UP
  - `1`: DOWN
  - `2`: LEFT
  - `3`: RIGHT

### Observation space
- Shape: `(4, 4)` by default (`size` is configurable)
- `observation_mode="raw"`: `int32` tile values (`0, 2, 4, ...`)
- `observation_mode="log2"`: `float32` (`2->1, 4->2, ...`, empty stays `0`)

### Reward configuration
- Built-in:
  - `reward_mode="score"`: merge points gained this step
  - `reward_mode="log_score"`: `log2(gained + 1.0)`
- Invalid move penalty:
  - `invalid_move_penalty` is applied for built-in modes when action does not change the board
- Custom reward:
  - pass `reward_fn(env, gained, invalid_move) -> float`
  - when `reward_fn` is provided, it overrides built-in reward mode behavior

Example custom reward usage:

```python
from rl2048.envs import Game2048Env
from rl2048.rewards import make_merge_space_max_tile_reward, merge_space_max_tile_reward

env = Game2048Env(reward_fn=merge_space_max_tile_reward)

tuned_reward = make_merge_space_max_tile_reward(
    empty_bonus_scale=0.2,
    max_tile_bonus_scale=0.02,
)
env = Game2048Env(reward_fn=tuned_reward)
```

### Episode termination
- `terminated=True` when no valid moves remain
- `truncated=True` when `max_steps` is set and reached

### Step info dictionary
Each `step()` returns info keys:
- `score`
- `gained`
- `max_tile`
- `empty_tiles`
- `invalid_move`
- `available_actions`
- `action_mask`
- `action_meanings`
- `steps`


## Scripts
Repository entry points for project workflows, including environment checks and
future training/evaluation utilities.

- `scripts/random_rollout.py`: quick random-agent rollout against the environment.
- Additional project scripts can be added under `scripts/` as training and
  evaluation work progresses.

## Tests
Automated validation for environment and board-logic correctness.

- `tests/test_logic.py`: board logic unit tests.
- `tests/test_env.py`: Gym environment behavior tests.
- Smoke run:

```bash
make smoke
```

- Run tests:

```bash
make test
```

- Optional lint:

```bash
make lint
```
