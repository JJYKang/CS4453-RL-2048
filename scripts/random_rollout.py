"""Run one random episode to sanity-check the environment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from rl2048.envs import Game2048Env

    env = Game2048Env(render_mode="ansi")
    obs, info = env.reset(seed=0)
    del obs

    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(env.render())
    print(
        f"Episode finished | score={info['score']} max_tile={info['max_tile']} total_reward={total_reward:.2f}"
    )


if __name__ == "__main__":
    main()
