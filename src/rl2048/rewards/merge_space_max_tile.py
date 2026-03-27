"""Merge + space + max-tile reward function for Game2048Env."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from rl2048.envs.game_2048_env import Game2048Env


def merge_space_max_tile_reward(
    env: Game2048Env,
    gained: int,
    invalid_move: bool,
    *,
    empty_bonus_scale: float = 0.1,
    max_tile_bonus_scale: float = 0.01,
) -> float:
    """Reward = merge score + empty-tile bonus + max-tile bonus."""
    empty_tiles = int(np.count_nonzero(env.board == 0))
    max_tile = int(np.max(env.board))
    reward = (
        float(gained)
        + empty_bonus_scale * float(empty_tiles)
        + max_tile_bonus_scale * float(np.log2(max(1, max_tile)))
    )
    if invalid_move:
        reward += env.invalid_move_penalty
    return reward


def make_merge_space_max_tile_reward(
    *,
    empty_bonus_scale: float = 0.1,
    max_tile_bonus_scale: float = 0.01,
) -> Callable[[Game2048Env, int, bool], float]:
    """Create a reward function with fixed shaping scales."""

    def reward(env: Game2048Env, gained: int, invalid_move: bool) -> float:
        return merge_space_max_tile_reward(
            env,
            gained,
            invalid_move,
            empty_bonus_scale=empty_bonus_scale,
            max_tile_bonus_scale=max_tile_bonus_scale,
        )

    return reward
