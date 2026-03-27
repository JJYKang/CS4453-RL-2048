from __future__ import annotations

import numpy as np
import pytest

from rl2048.envs import Game2048Env
from rl2048.rewards import make_merge_space_max_tile_reward, merge_space_max_tile_reward


def test_reset_creates_two_start_tiles() -> None:
    env = Game2048Env()
    obs, info = env.reset(seed=123)

    assert obs.shape == (4, 4)
    assert np.count_nonzero(env.board) == 2
    assert info["score"] == 0


def test_step_returns_gymnasium_tuple() -> None:
    env = Game2048Env(max_steps=1)
    env.reset(seed=5)

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (4, 4)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert truncated is True
    assert info["action_mask"].shape == (4,)


def test_invalid_move_penalty_applied() -> None:
    env = Game2048Env(invalid_move_penalty=-1.5, reward_mode="score")
    env.reset(seed=7)

    env.board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [2, 4, 8, 16],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward, _, _, info = env.step(0)  # UP is invalid on this board.

    assert info["invalid_move"] is True
    assert reward == -1.5


def test_custom_reward_function_overrides_builtin_behavior() -> None:
    def reward_fn(_env: Game2048Env, gained: int, invalid_move: bool) -> float:
        return float(gained + 1000 if invalid_move else gained + 10)

    env = Game2048Env(
        reward_mode="score",
        invalid_move_penalty=-999.0,
        reward_fn=reward_fn,
    )
    env.reset(seed=7)

    env.board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [2, 4, 8, 16],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward, _, _, info = env.step(0)  # UP is invalid on this board.

    assert info["invalid_move"] is True
    assert reward == 1000.0


def test_unsupported_reward_mode_rejected_without_custom_fn() -> None:
    with pytest.raises(ValueError, match="reward_mode must be one of"):
        Game2048Env(reward_mode="shaped")


def test_merge_space_max_tile_reward_works_as_custom_fn() -> None:
    env = Game2048Env(reward_fn=merge_space_max_tile_reward)
    env.reset(seed=3)

    _, reward, _, _, _ = env.step(0)

    assert env.reward_mode == "custom"
    assert isinstance(reward, float)


def test_merge_space_max_tile_reward_factory_works() -> None:
    reward_fn = make_merge_space_max_tile_reward(
        empty_bonus_scale=0.2,
        max_tile_bonus_scale=0.02,
    )
    env = Game2048Env(reward_fn=reward_fn)
    env.reset(seed=11)

    _, reward, _, _, _ = env.step(0)

    assert isinstance(reward, float)
