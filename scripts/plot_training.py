"""Plot DQN training progress in the style of the reference graph.

Three overlaid metrics:
  - Green bars:  Max tile reached per episode
  - Blue line:   Mean steps over last N episodes (left y-axis)
  - Red line:    Mean total reward over last N episodes (right y-axis)

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --log runs/dqn_checkpoints/episode_log.csv
    python scripts/plot_training.py --window 50 --out training_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "runs" / "dqn_checkpoints" / "episode_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG, help="Path to episode_log.csv",
    )
    parser.add_argument(
        "--window", type=int, default=50, help="Rolling average window size",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Save figure to file instead of showing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.log)
    w = args.window

    episodes = df["episode"].values
    rewards = df["reward"].values
    max_tiles = df["max_tile"].values
    steps = df["steps"].values

    # Rolling averages
    reward_rolling = pd.Series(rewards).rolling(window=w, min_periods=1).mean().values
    steps_rolling = pd.Series(steps).rolling(window=w, min_periods=1).mean().values

    # --- Plot ---
    fig, ax_left = plt.subplots(figsize=(16, 7))
    ax_right = ax_left.twinx()

    # Green bars: max tile per episode (behind everything)
    ax_left.bar(
        episodes, max_tiles,
        width=1.0,
        color="#81C784",
        alpha=0.4,
        label="Max cell value seen on board",
        zorder=1,
    )

    # Blue line: mean steps (left axis)
    ax_left.plot(
        episodes, steps_rolling,
        color="#1565C0",
        linewidth=1.0,
        label=f"Mean steps over last {w} episodes",
        zorder=3,
    )

    # Red line: mean total reward (right axis)
    ax_right.plot(
        episodes, reward_rolling,
        color="#D32F2F",
        linewidth=0.8,
        label=f"Mean total rewards over last {w} episodes",
        zorder=2,
    )

    # --- Axis labels ---
    ax_left.set_xlabel("Episode", fontsize=12)
    ax_left.set_ylabel("Steps / Max Tile", fontsize=12)
    ax_right.set_ylabel("Mean Total Reward", fontsize=12)

    # --- Combined legend ---
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
    )

    # --- Styling ---
    ax_left.set_xlim(0, episodes[-1])
    ax_left.grid(True, alpha=0.2)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
