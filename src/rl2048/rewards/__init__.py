"""Reusable reward functions for 2048 RL experiments."""

from rl2048.rewards.merge_space_max_tile import (
    make_merge_space_max_tile_reward,
    merge_space_max_tile_reward,
)

__all__ = ["merge_space_max_tile_reward", "make_merge_space_max_tile_reward"]
