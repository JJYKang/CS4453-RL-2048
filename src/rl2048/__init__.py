"""Top-level package for the 2048 RL project."""

from rl2048.envs import ACTION_MEANINGS, Game2048Env, register_env
from rl2048.rewards import make_merge_space_max_tile_reward, merge_space_max_tile_reward
from rl2048.rewards import make_corner_shape_reward, corner_shape_reward

register_env()

__all__ = [
    "Game2048Env",
    "ACTION_MEANINGS",
    "register_env",
    "merge_space_max_tile_reward",
    "make_merge_space_max_tile_reward",
    "corner_shape_reward",
    "make_corner_shape_reward",
]
