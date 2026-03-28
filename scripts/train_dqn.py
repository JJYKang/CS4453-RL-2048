"""Train DQN against the 2048 environment using a YAML config file."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dqn.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Config at {path} must be a YAML mapping.")
    return config


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess(
    obs: np.ndarray,
    *,
    use_cnn: bool,
    max_log2_value: float,
    one_hot_encode: Any,
) -> np.ndarray:
    if use_cnn:
        raw = np.where(obs > 0, 2**obs, 0).astype(np.int64)
        return one_hot_encode(raw)
    return obs.flatten() / max_log2_value


def train(config: dict[str, Any]) -> None:
    from rl2048.dqn import DQNAgent, DQN_CNN, DQN_MLP
    from rl2048.dqn.preprocess import one_hot_encode
    from rl2048.envs import Game2048Env

    training_cfg = config.get("training", {})
    env_cfg = config.get("env", {})
    network_cfg = config.get("network", {})
    agent_cfg = config.get("agent", {})

    use_cnn = bool(training_cfg.get("use_cnn", False))
    num_episodes = int(training_cfg.get("num_episodes", 5000))
    log_every = int(training_cfg.get("log_every", 100))
    save_every = int(training_cfg.get("save_every", 1000))
    seed = training_cfg.get("seed")
    max_log2_value = float(training_cfg.get("max_log2_value", 17.0))

    checkpoint_dir = PROJECT_ROOT / str(
        training_cfg.get("checkpoint_dir", "runs/dqn_checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_name = str(training_cfg.get("final_model_name", "final_model.pt"))

    env = Game2048Env(
        observation_mode=str(env_cfg.get("observation_mode", "log2")),
        reward_mode=str(env_cfg.get("reward_mode", "score")),
        invalid_move_penalty=float(env_cfg.get("invalid_move_penalty", -1.0)),
    )

    n_actions = int(network_cfg.get("n_actions", 4))
    if use_cnn:
        in_channels = int(network_cfg.get("cnn_channels", 16))
        q_net = DQN_CNN(in_channels=in_channels, n_actions=n_actions)
        target_net = DQN_CNN(in_channels=in_channels, n_actions=n_actions)
    else:
        input_dim = int(network_cfg.get("mlp_input_dim", 16))
        hidden = int(network_cfg.get("mlp_hidden_size", 256))
        q_net = DQN_MLP(input_dim=input_dim, hidden=hidden, n_actions=n_actions)
        target_net = DQN_MLP(input_dim=input_dim, hidden=hidden, n_actions=n_actions)

    device = get_device()
    agent = DQNAgent(
        q_network=q_net,
        target_network=target_net,
        lr=float(agent_cfg.get("lr", 5e-4)),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        buffer_capacity=int(agent_cfg.get("buffer_capacity", 100_000)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 1_000)),
        eps_start=float(agent_cfg.get("eps_start", 1.0)),
        eps_end=float(agent_cfg.get("eps_end", 0.01)),
        eps_decay_steps=int(agent_cfg.get("eps_decay_steps", 100_000)),
        device=device,
    )

    print(f"Config: {config}")
    print(f"Device: {device}")
    print(f"Network: {'CNN' if use_cnn else 'MLP'}")
    print(f"Params: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"Training for {num_episodes} episodes...\n")

    reward_window = deque(maxlen=log_every)
    max_tile_window = deque(maxlen=log_every)
    loss_window = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        reset_seed = int(seed) + episode - 1 if seed is not None else None
        obs, info = env.reset(seed=reset_seed)
        state = preprocess(
            obs,
            use_cnn=use_cnn,
            max_log2_value=max_log2_value,
            one_hot_encode=one_hot_encode,
        )
        episode_reward = 0.0
        done = False

        while not done:
            valid_actions = list(info["available_actions"])
            action = agent.select_action(state, valid_actions)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess(
                next_obs,
                use_cnn=use_cnn,
                max_log2_value=max_log2_value,
                one_hot_encode=one_hot_encode,
            )
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                loss_window.append(loss)

            state = next_state
            episode_reward += reward

        reward_window.append(episode_reward)
        max_tile_window.append(info["max_tile"])

        if episode % log_every == 0:
            avg_reward = float(np.mean(reward_window))
            avg_loss = float(np.mean(loss_window)) if loss_window else 0.0
            max_tile = max(max_tile_window)
            print(
                f"Ep {episode:>6d} | "
                f"Avg Reward: {avg_reward:>8.1f} | "
                f"Max Tile: {max_tile:>5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):,}"
            )

        if episode % save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode}.pt"
            agent.save(str(checkpoint_path))

    final_model_path = checkpoint_dir / final_model_name
    agent.save(str(final_model_path))
    env.close()
    print("\nTraining complete!")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
