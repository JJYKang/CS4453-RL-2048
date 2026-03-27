"""Environment exports and registration helpers."""

from gymnasium import error
from gymnasium.envs.registration import register

from rl2048.envs.game_2048_env import ACTION_MEANINGS, Game2048Env
from rl2048.envs.types import InfoDict, Obs, RewardFn

ENV_ID = "Game2048-v0"


def register_env() -> None:
    """Register the env once so `gymnasium.make(ENV_ID)` works."""
    try:
        register(id=ENV_ID, entry_point="rl2048.envs:Game2048Env")
    except error.Error:
        # Already registered.
        pass


__all__ = [
    "ACTION_MEANINGS",
    "Game2048Env",
    "ENV_ID",
    "InfoDict",
    "Obs",
    "RewardFn",
    "register_env",
]
