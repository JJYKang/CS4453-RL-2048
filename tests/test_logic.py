from __future__ import annotations

import numpy as np

from rl2048.envs.logic import DOWN, LEFT, RIGHT, UP, available_actions, has_valid_moves, move


def test_move_left_merges_once_per_pair() -> None:
    board = np.array(
        [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    moved, gained, changed = move(board, LEFT)

    assert changed is True
    assert gained == 4
    np.testing.assert_array_equal(moved[0], np.array([4, 2, 0, 0], dtype=np.int32))


def test_move_up_merges_column() -> None:
    board = np.array(
        [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    moved, gained, changed = move(board, UP)

    assert changed is True
    assert gained == 4
    assert moved[0, 0] == 4
    assert moved[1, 0] == 0


def test_move_right_merges_once_per_pair() -> None:
    board = np.array(
        [
            [0, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    moved, gained, changed = move(board, RIGHT)

    assert changed is True
    assert gained == 4
    np.testing.assert_array_equal(moved[0], np.array([0, 0, 2, 4], dtype=np.int32))


def test_move_down_merges_once_per_pair() -> None:
    board = np.array(
        [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    moved, gained, changed = move(board, DOWN)

    assert changed is True
    assert gained == 4
    np.testing.assert_array_equal(moved[:, 0], np.array([0, 0, 2, 4], dtype=np.int32))


def test_has_valid_moves_false_for_locked_board() -> None:
    board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )

    assert has_valid_moves(board) is False


def test_available_actions_returns_subset() -> None:
    board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    actions = set(available_actions(board))

    assert DOWN in actions
    assert UP not in actions
