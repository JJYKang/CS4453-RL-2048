"""
Reward for 2048 incorporating corner anchoring and board-structure shaping.

Includes:
- merge reward as the main signal
- reward for creating empty spaces
- bonus for keeping the max tile in a fixed corner
- penalty if the max tile leaves that corner
- top-row monotonicity (decreasing from left to right) signal toward that corner
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rl2048.envs.game_2048_env import Game2048Env


def _corner_position(corner: str, size: int) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, size - 1
    if corner == "bottom_left":
        return size - 1, 0
    if corner == "bottom_right":
        return size - 1, size - 1
    raise ValueError(
        "corner must be one of "
        "{'top_left', 'top_right', 'bottom_left', 'bottom_right'}"
    )


def _max_tile_in_corner(board: np.ndarray, corner: str) -> bool:
    row, col = _corner_position(corner, board.shape[0])
    max_tile = int(np.max(board))
    return max_tile > 0 and int(board[row, col]) == max_tile


def _log_board(board: np.ndarray) -> np.ndarray:
    out = np.zeros_like(board, dtype=np.float32)
    non_zero = board > 0
    out[non_zero] = np.log2(board[non_zero]).astype(np.float32)
    return out


def _oriented_log_board(board: np.ndarray, corner: str) -> np.ndarray:
    logged = _log_board(board)

    if corner == "top_left":
        return logged
    if corner == "top_right":
        return np.fliplr(logged)
    if corner == "bottom_left":
        return np.flipud(logged)
    if corner == "bottom_right":
        return np.flipud(np.fliplr(logged))

    raise ValueError(
        "corner must be one of "
        "{'top_left', 'top_right', 'bottom_left', 'bottom_right'}"
    )


def _top_row_monotonicity_penalty(board: np.ndarray, corner: str) -> float:
    """
    Lower is better.

    After orienting the board toward the chosen corner, the top row should
    decrease from left to right. Only violations are penalized.
    """
    oriented = _oriented_log_board(board, corner)
    top_row = oriented[0]
    violations = np.maximum(0.0, top_row[1:] - top_row[:-1])
    return float(np.sum(violations))


def corner_shape_reward(
    env: Game2048Env,
    gained: int,
    invalid_move: bool,
    *,
    corner: str = "top_left",
    merge_log_scale: float = 1.0,
    empty_delta_scale: float = 0.4,
    corner_bonus_scale: float = 1.0,
    leave_corner_penalty: float = 3.0,
    top_row_monotonicity_delta_scale: float = 0.4,
) -> float:
    """
    Reward with binary corner anchoring and minimal board-structure shaping.

    Terms:
    - log2(1 + gained)
    - delta in empty tiles
    - bonus when the max tile is in the chosen corner
    - penalty when the max tile leaves that corner
    - improvement in top-row monotonicity toward the chosen corner
    - invalid move penalty from the environment
    """
    prev_board = env.prev_board
    curr_board = env.board

    merge_term = merge_log_scale * float(np.log2(gained + 1.0))

    prev_empty = int(np.count_nonzero(prev_board == 0))
    curr_empty = int(np.count_nonzero(curr_board == 0))
    empty_delta_term = empty_delta_scale * float(curr_empty - prev_empty)

    prev_max_in_corner = _max_tile_in_corner(prev_board, corner)
    curr_max_in_corner = _max_tile_in_corner(curr_board, corner)

    corner_term = 0.0
    if curr_max_in_corner:
        corner_term += corner_bonus_scale
    if prev_max_in_corner and not curr_max_in_corner:
        corner_term -= leave_corner_penalty

    prev_top_row_penalty = _top_row_monotonicity_penalty(prev_board, corner)
    curr_top_row_penalty = _top_row_monotonicity_penalty(curr_board, corner)
    top_row_monotonicity_term = top_row_monotonicity_delta_scale * float(
        prev_top_row_penalty - curr_top_row_penalty
    )

    reward = (
        merge_term
        + empty_delta_term
        + corner_term
        + top_row_monotonicity_term
    )

    if invalid_move:
        reward += env.invalid_move_penalty

    return reward


def make_corner_shape_reward(
    *,
    corner: str = "top_left",
    merge_log_scale: float = 1.0,
    empty_delta_scale: float = 0.4,
    corner_bonus_scale: float = 1.0,
    leave_corner_penalty: float = 3.0,
    top_row_monotonicity_delta_scale: float = 0.4,
) -> Callable[[Game2048Env, int, bool], float]:
    """Create a reward function with fixed shaping scales."""

    def reward(env: Game2048Env, gained: int, invalid_move: bool) -> float:
        return corner_shape_reward(
            env,
            gained,
            invalid_move,
            corner=corner,
            merge_log_scale=merge_log_scale,
            empty_delta_scale=empty_delta_scale,
            corner_bonus_scale=corner_bonus_scale,
            leave_corner_penalty=leave_corner_penalty,
            top_row_monotonicity_delta_scale=top_row_monotonicity_delta_scale,
        )

    return reward