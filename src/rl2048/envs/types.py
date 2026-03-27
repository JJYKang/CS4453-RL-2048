"""Shared type definitions for 2048 environment modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rl2048.envs.game_2048_env import Game2048Env

Board = NDArray[np.int32]
Obs = NDArray[np.int32] | NDArray[np.float32]
RewardFn = Callable[["Game2048Env", int, bool], float]


class InfoDict(TypedDict):
    """Per-step diagnostic information returned by ``reset`` and ``step``."""

    score: int
    gained: int
    max_tile: int
    empty_tiles: int
    invalid_move: bool
    available_actions: Tuple[int, ...]
    action_mask: NDArray[np.int8]
    action_meanings: Dict[int, str]
    steps: int
