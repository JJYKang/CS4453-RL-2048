"""Train a CNN-based DQN agent on 2048 using the corner-shaped reward function.

Network:   DQN_CNN  (one-hot encoded board, multi-scale ConvBlocks)
Reward:    corner_shape_reward  (merge + empty-delta + corner anchor + monotonicity)
Config:    configs/dqn_cnn_corner.yaml  (all hyperparameters live there)

Usage
-----
    python scripts/train_dqn_cnn_corner.py
    python scripts/train_dqn_cnn_corner.py --config configs/dqn_cnn_corner.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup — allows running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dqn_cnn_corner.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file (default: configs/dqn_cnn_corner.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Config at {path} must be a YAML mapping.")
    return config


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config: dict[str, Any]) -> None:
    from rl2048.dqn import DQN_CNN, DQNAgent
    from rl2048.dqn.preprocess import one_hot_encode
    from rl2048.envs import Game2048Env
    from rl2048.rewards import make_corner_shape_reward

    # ---- Unpack config sections ----------------------------------------
    training_cfg = config.get("training", {})
    env_cfg      = config.get("env", {})
    network_cfg  = config.get("network", {})
    agent_cfg    = config.get("agent", {})
    reward_cfg   = config.get("reward", {})

    # ---- Training meta -------------------------------------------------
    num_episodes     = int(training_cfg.get("num_episodes", 10_000))
    log_every        = int(training_cfg.get("log_every", 100))
    save_every       = int(training_cfg.get("save_every", 1_000))
    seed             = training_cfg.get("seed")
    max_log2_value   = float(training_cfg.get("max_log2_value", 17.0))
    checkpoint_dir   = PROJECT_ROOT / str(training_cfg.get("checkpoint_dir", "runs/dqn_cnn_corner"))
    final_model_name = str(training_cfg.get("final_model_name", "final_model.pt"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "episode_log.csv"

    # ---- Corner-shaped reward function ---------------------------------
    corner_reward_fn = make_corner_shape_reward(
        corner=str(reward_cfg.get("corner", "top_left")),
        merge_log_scale=float(reward_cfg.get("merge_log_scale", 1.0)),
        empty_delta_scale=float(reward_cfg.get("empty_delta_scale", 0.4)),
        corner_bonus_scale=float(reward_cfg.get("corner_bonus_scale", 1.5)),
        leave_corner_penalty=float(reward_cfg.get("leave_corner_penalty", 5.0)),
        top_row_monotonicity_delta_scale=float(
            reward_cfg.get("top_row_monotonicity_delta_scale", 0.5)
        ),
    )

    # ---- Environment ---------------------------------------------------
    env = Game2048Env(
        observation_mode=str(env_cfg.get("observation_mode", "log2")),
        # reward_mode is overridden by reward_fn → pass a dummy string
        reward_mode="score",
        reward_fn=corner_reward_fn,
        invalid_move_penalty=float(env_cfg.get("invalid_move_penalty", -5.0)),
    )

    # ---- Network -------------------------------------------------------
    in_channels = int(network_cfg.get("cnn_channels", 16))
    n_actions   = int(network_cfg.get("n_actions", 4))
    q_net       = DQN_CNN(in_channels=in_channels, n_actions=n_actions)
    target_net  = DQN_CNN(in_channels=in_channels, n_actions=n_actions)

    # ---- Agent ---------------------------------------------------------
    device = get_device()
    agent = DQNAgent(
        q_network=q_net,
        target_network=target_net,
        lr=float(agent_cfg.get("lr", 3e-4)),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        buffer_capacity=int(agent_cfg.get("buffer_capacity", 100_000)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 1_000)),
        eps_start=float(agent_cfg.get("eps_start", 1.0)),
        eps_end=float(agent_cfg.get("eps_end", 0.01)),
        eps_decay_steps=int(agent_cfg.get("eps_decay_steps", 200_000)),
        device=device,
    )

    # ---- Logging -------------------------------------------------------
    print("=" * 65)
    print(f"  CNN-DQN + Corner-Shaped Reward")
    print(f"  Corner anchor : {reward_cfg.get('corner', 'top_left')}")
    print(f"  Device        : {device}")
    print(f"  Parameters    : {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"  Episodes      : {num_episodes:,}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 65)
    print()

    log_file   = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["episode", "reward", "max_tile", "steps", "loss", "epsilon"])

    reward_window   = deque(maxlen=log_every)
    max_tile_window = deque(maxlen=log_every)
    loss_window     = deque(maxlen=100)

    # ---- Episode loop --------------------------------------------------
    for episode in range(1, num_episodes + 1):
        reset_seed = int(seed) + episode - 1 if seed is not None else None
        obs, info  = env.reset(seed=reset_seed)

        # CNN preprocessing: log2 obs → raw tile values → one-hot (C×4×4)
        raw_board  = np.where(obs > 0, 2 ** obs, 0).astype(np.int64)
        state      = one_hot_encode(raw_board)

        episode_reward = 0.0
        done           = False

        while not done:
            valid_actions = list(info["available_actions"])
            action        = agent.select_action(state, valid_actions)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Preprocess next state
            next_raw   = np.where(next_obs > 0, 2 ** next_obs, 0).astype(np.int64)
            next_state = one_hot_encode(next_raw)

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                loss_window.append(loss)

            state          = next_state
            episode_reward += reward

        # ---- Per-episode bookkeeping -----------------------------------
        reward_window.append(episode_reward)
        max_tile_window.append(info["max_tile"])

        avg_loss = float(np.mean(loss_window)) if loss_window else 0.0
        csv_writer.writerow([
            episode,
            f"{episode_reward:.2f}",
            info["max_tile"],
            info["steps"],
            f"{avg_loss:.4f}",
            f"{agent.epsilon:.4f}",
        ])

        if episode % log_every == 0:
            log_file.flush()
            avg_reward = float(np.mean(reward_window))
            print(
                f"Ep {episode:>6d} | "
                f"Avg Reward: {avg_reward:>9.2f} | "
                f"Max Tile: {max(max_tile_window):>5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):,}"
            )

        if episode % save_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_ep{episode}.pt"
            agent.save(str(ckpt_path))

    # ---- Save final model ----------------------------------------------
    final_path = checkpoint_dir / final_model_name
    agent.save(str(final_path))
    log_file.close()
    env.close()
    print("\nTraining complete!")
    print(f"Final model saved to {final_path}")
    print(f"Episode log saved to  {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
