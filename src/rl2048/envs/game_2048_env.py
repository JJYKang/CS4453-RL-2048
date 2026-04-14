"""Gymnasium-compatible 2048 environment.

This module provides :class:`Game2048Env`, a 2048 implementation that follows
the Gymnasium API:
- ``reset() -> (observation, info)``
- ``step(action) -> (observation, reward, terminated, truncated, info)``

Environment highlights:
- Action space: ``Discrete(4)`` with ``0=UP, 1=DOWN, 2=LEFT, 3=RIGHT``.
- Observation modes:
  - ``raw``: board tile values as ``int32``.
  - ``log2``: ``log2(tile)`` as ``float32`` (empty cells remain 0).
- Reward modes:
  - ``score`` and ``log_score`` built-ins.
  - ``reward_fn(env, gained, invalid_move)`` for custom reward logic.
- Termination: no legal moves remain.
- Truncation: ``max_steps`` limit (when configured).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from rl2048.envs.logic import (
    ACTION_MEANINGS,
    action_mask,
    create_start_board,
    move,
    spawn_random_tile,
)
from rl2048.envs.types import Board, InfoDict, Obs, RewardFn


class Game2048Env(gym.Env):
    """2048 environment for RL experiments.

    State is represented by a square board (default 4x4). At each step the
    agent chooses one of four slide directions. Legal moves shift/merge tiles,
    then spawn a new tile (2 with ``spawn_probability_2`` else 4).

    Rewards are computed from merge gain and optional invalid-move penalty using
    either built-in modes (``score``, ``log_score``) or a custom ``reward_fn``.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 4,
        spawn_probability_2: float = 0.9,
        invalid_move_penalty: float = -1.0,
        max_steps: Optional[int] = None,
        observation_mode: str = "log2",
        reward_mode: str = "score",
        reward_fn: Optional[RewardFn] = None,
        max_tile_value: int = 131072,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create a 2048 environment.

        Args:
            size: Board width/height. Standard 2048 uses ``4``.
            spawn_probability_2: Probability that a spawned tile is ``2``
                (otherwise ``4``).
            invalid_move_penalty: Added to reward for actions that do not
                change the board.
            max_steps: Optional step cap per episode. When reached, episode is
                truncated.
            observation_mode: How observations are encoded.
                - "raw": board contains tile values as int32 (0, 2, 4, ...).
                - "log2": board contains log2(tile) as float32 (0 stays 0).
            reward_mode: How per-step reward is computed.
                - "score": raw merge points gained this step.
                - "log_score": log2(gained + 1.0) to compress large merges.
            reward_fn: Optional custom reward callable with signature
                ``(env, gained, invalid_move) -> float``.
                When provided, it overrides ``reward_mode`` and the built-in
                invalid-move penalty behavior.
            max_tile_value: Upper bound used in observation space definition.
            render_mode: Output format used by :meth:`render`.
                - None: rendering disabled (render returns None).
                - "human": board is printed to stdout.
                - "ansi": board is returned as a string.
            seed: Optional default seed used on first ``reset(seed=None)``.
        """
        super().__init__()

        if size < 2:
            raise ValueError("size must be >= 2")
        if not 0.0 <= spawn_probability_2 <= 1.0:
            raise ValueError("spawn_probability_2 must be in [0, 1]")
        if observation_mode not in {"raw", "log2"}:
            raise ValueError("observation_mode must be one of {'raw', 'log2'}")
        if reward_mode not in {"score", "log_score"} and reward_fn is None:
            raise ValueError("reward_mode must be one of {'score', 'log_score'}")
        if reward_fn is not None and not callable(reward_fn):
            raise TypeError("reward_fn must be callable")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        self.size = size
        self.spawn_probability_2 = spawn_probability_2
        self.invalid_move_penalty = invalid_move_penalty
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.reward_mode = "custom" if reward_fn is not None else reward_mode
        self._custom_reward_fn = reward_fn
        self.max_tile_value = max_tile_value
        self.render_mode = render_mode
        self._reward_fn: Callable[[int, bool], float]
        if reward_fn is not None:
            self._reward_fn = self._reward_custom
        else:
            self._reward_fn = {
                "score": self._reward_score,
                "log_score": self._reward_log_score,
            }[reward_mode]

        self.action_space = spaces.Discrete(4)
        if observation_mode == "raw":
            self.observation_space = spaces.Box(
                low=0,
                high=max_tile_value,
                shape=(size, size),
                dtype=np.int32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=float(np.log2(max_tile_value)),
                shape=(size, size),
                dtype=np.float32,
            )

        self._seed_on_first_reset = seed
        self._seed_used = False

        self.board: Board = np.zeros((self.size, self.size), dtype=np.int32)
        self.prev_board: Board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.steps = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Obs, InfoDict]:
        """Reset environment state and return initial observation/info.

        Args:
            seed: Optional seed for Gymnasium RNG reinitialization.
            options: Unused Gymnasium reset options (accepted for API
                compatibility).

        Returns:
            A tuple ``(observation, info)``.
            - ``observation`` is the agent-facing state tensor based on
              ``observation_mode``:
              - ``raw``: tile values (int32)
              - ``log2``: log2-scaled values (float32)
            - ``info`` is diagnostics metadata (not the primary learning
              signal), including fields such as ``score``, ``max_tile``,
              ``invalid_move``, ``available_actions``, and ``action_mask``.
        """
        del options

        if seed is None and not self._seed_used:
            seed = self._seed_on_first_reset

        super().reset(seed=seed)
        self._seed_used = True

        self.board = create_start_board(
            size=self.size,
            rng=self.np_random,
            probability_2=self.spawn_probability_2,
        )
        self.prev_board = self.board.copy()
        self.score = 0
        self.steps = 0

        mask = action_mask(self.board)
        available = tuple(int(action) for action in np.flatnonzero(mask))
        return self._get_observation(), self._build_info(
            gained=0,
            invalid_move=False,
            available=available,
            mask=mask,
        )

    def step(self, action: int) -> Tuple[Obs, float, bool, bool, InfoDict]:
        """Apply one action and advance the environment by one transition.

        Args:
            action: Integer action in ``{0, 1, 2, 3}`` corresponding to
                ``UP, DOWN, LEFT, RIGHT``.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` following
            Gymnasium conventions.
            - ``observation`` is the next agent-facing state tensor.
            - ``terminated`` is True when no valid moves remain.
            - ``truncated`` is True when ``max_steps`` is reached.
            - ``info`` is a typed diagnostics dictionary (`InfoDict`) with
              score/move metadata useful for logging and analysis.
        """
        if isinstance(action, np.ndarray):
            action = int(action.item())
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}.")

        self.steps += 1
        self.prev_board = self.board.copy()

        moved_board, gained, changed = move(self.board, int(action))
        invalid_move = not changed

        if changed:
            self.board = moved_board
            spawn_random_tile(
                self.board,
                rng=self.np_random,
                probability_2=self.spawn_probability_2,
            )

        self.score += int(gained)
        reward = self._compute_reward(gained=gained, invalid_move=invalid_move)

        mask = action_mask(self.board)
        available = tuple(int(action) for action in np.flatnonzero(mask))
        terminated = not bool(mask.any())
        truncated = self.max_steps is not None and self.steps >= self.max_steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(
                gained=gained,
                invalid_move=invalid_move,
                available=available,
                mask=mask,
            ),
        )

    def render(self) -> Optional[str]:
        """Render the board according to ``render_mode``.

        Returns:
            - ``None`` for ``render_mode=None`` and ``render_mode='human'``.
            - String board representation for ``render_mode='ansi'``.
        """
        board_str = self._board_to_string()
        if self.render_mode == "human":
            print(board_str)
            return None
        if self.render_mode == "ansi":
            return board_str
        return None

    def close(self) -> None:
        """Release environment resources.

        No external resources are held for this environment, so this is a
        no-op provided for API completeness.
        """
        pass

    def _get_observation(self) -> Obs:
        if self.observation_mode == "raw":
            return self.board.copy()

        obs = np.zeros_like(self.board, dtype=np.float32)
        non_zero = self.board > 0
        obs[non_zero] = np.log2(self.board[non_zero]).astype(np.float32)
        return obs

    def _compute_reward(self, gained: int, invalid_move: bool) -> float:
        return self._reward_fn(gained, invalid_move)

    def _reward_score(self, gained: int, invalid_move: bool) -> float:
        reward = float(gained)
        if invalid_move:
            reward += self.invalid_move_penalty
        return reward

    def _reward_log_score(self, gained: int, invalid_move: bool) -> float:
        reward = float(np.log2(gained + 1.0))
        if invalid_move:
            reward += self.invalid_move_penalty
        return reward

    def _reward_custom(self, gained: int, invalid_move: bool) -> float:
        if self._custom_reward_fn is None:
            raise RuntimeError("Custom reward function is not set.")
        return float(self._custom_reward_fn(self, gained, invalid_move))

    def _build_info(
        self,
        gained: int,
        invalid_move: bool,
        available: Tuple[int, ...],
        mask: NDArray[np.int8],
    ) -> InfoDict:
        return {
            "score": int(self.score),
            "gained": int(gained),
            "max_tile": int(np.max(self.board)),
            "empty_tiles": int(np.count_nonzero(self.board == 0)),
            "invalid_move": bool(invalid_move),
            "available_actions": available,
            "action_mask": mask,
            "action_meanings": ACTION_MEANINGS,
            "steps": int(self.steps),
        }

    def _board_to_string(self) -> str:
        max_value = int(np.max(self.board))
        width = max(4, len(str(max_value)))
        sep = "+" + "+".join(["-" * (width + 2)] * self.size) + "+"

        lines = [f"score={self.score} steps={self.steps} max_tile={max_value}", sep]
        for row in self.board:
            cells = []
            for value in row:
                text = "." if value == 0 else str(int(value))
                cells.append(f" {text:>{width}} ")
            lines.append("|" + "|".join(cells) + "|")
            lines.append(sep)
        return "\n".join(lines)
