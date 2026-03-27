"""Pure 2048 board logic used by the RL environment."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from rl2048.envs.types import Board

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_MEANINGS = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}


def _move_row_left(row: Board) -> Tuple[Board, int]:
    """Slide one row left and merge adjacent equal values once."""
    non_zero = row[row != 0]
    merged_values = []
    gained = 0

    idx = 0
    while idx < non_zero.size:
        value = int(non_zero[idx])
        if idx + 1 < non_zero.size and value == int(non_zero[idx + 1]):
            value *= 2
            gained += value
            idx += 2
        else:
            idx += 1
        merged_values.append(value)

    out = np.zeros_like(row)
    if merged_values:
        out[: len(merged_values)] = np.asarray(merged_values, dtype=np.int32)
    return out, gained


def _move_left(board: Board) -> Tuple[Board, int, bool]:
    """Apply a LEFT move to the full board."""
    moved = np.zeros_like(board)
    gained = 0

    for ridx, row in enumerate(board):
        new_row, row_gain = _move_row_left(row)
        moved[ridx] = new_row
        gained += row_gain

    changed = not np.array_equal(moved, board)
    return moved, gained, changed


def _restore_identity(board: Board) -> Board:
    return board


def _restore_transpose(board: Board) -> Board:
    return board.T


def _restore_fliplr_transpose(board: Board) -> Board:
    return np.fliplr(board).T


def move(board: Board, action: int) -> Tuple[Board, int, bool]:
    """Apply action to a board and return (new_board, gained_score, changed)."""
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("Board must be a square 2D array.")
    if action not in ACTIONS:
        raise ValueError(f"Invalid action {action}. Expected one of {ACTIONS}.")

    restore: Callable[[Board], Board]
    if action == LEFT:
        transformed = board
        restore = _restore_identity
    elif action == RIGHT:
        transformed = np.fliplr(board)
        restore = np.fliplr
    elif action == UP:
        transformed = board.T
        restore = _restore_transpose
    else:  # DOWN
        transformed = np.fliplr(board.T)
        restore = _restore_fliplr_transpose

    moved, gained, _ = _move_left(transformed)
    restored = restore(moved).astype(np.int32, copy=False)
    changed = not np.array_equal(restored, board)
    return restored, gained, changed


def spawn_random_tile(
    board: Board,
    rng: np.random.Generator,
    probability_2: float = 0.9,
) -> bool:
    """Spawn a 2 or 4 in a random empty cell. Returns False if board is full."""
    empty = np.argwhere(board == 0)
    if empty.size == 0:
        return False

    idx = int(rng.integers(0, len(empty)))
    row, col = empty[idx]
    board[row, col] = 2 if float(rng.random()) < probability_2 else 4
    return True


def create_start_board(
    size: int,
    rng: np.random.Generator,
    probability_2: float = 0.9,
) -> Board:
    """Create a fresh board with two starting tiles."""
    board = np.zeros((size, size), dtype=np.int32)
    spawn_random_tile(board, rng, probability_2)
    spawn_random_tile(board, rng, probability_2)
    return board


def has_valid_moves(board: Board) -> bool:
    """Check whether at least one move is legal from this board state."""
    if np.any(board == 0):
        return True

    if np.any(board[:, :-1] == board[:, 1:]):
        return True
    if np.any(board[:-1, :] == board[1:, :]):
        return True

    return False


def available_actions(board: Board) -> Tuple[int, ...]:
    """Return all actions that produce a state change."""
    valid = []
    for action in ACTIONS:
        _, _, changed = move(board, action)
        if changed:
            valid.append(action)
    return tuple(valid)


def action_mask(board: Board) -> NDArray[np.int8]:
    """Return binary mask in action order [UP, DOWN, LEFT, RIGHT]."""
    mask = np.zeros(len(ACTIONS), dtype=np.int8)
    for action in available_actions(board):
        mask[action] = 1
    return mask
