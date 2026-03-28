"""
Training loop for DQN 2048.

Usage:
    pip install -e path/to/CS4453-RL-2048   # install your teammate's env
    python train.py

Integrates with JJYKang's Game2048Env (Gymnasium API).
"""

import numpy as np
import torch
from collections import deque

from dqn import DQNAgent, DQN_MLP, DQN_CNN
from dqn.preprocess import log2_preprocess, one_hot_encode

# ======================================================================
# Import the real environment
# ======================================================================
# Option A: if you've pip-installed your teammate's package
from rl2048.envs.game_2048_env import Game2048Env

# Option B: if you just copied the file locally, adjust the import path
# from game_2048_env import Game2048Env


# ======================================================================
# Config
# ======================================================================
USE_CNN = False  # Set True to use DQN_CNN + one-hot encoding

HYPERPARAMS = {
    "lr": 5e-4,
    "gamma": 0.99,
    "buffer_capacity": 100_000,
    "batch_size": 64,
    "target_update_freq": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay_steps": 100_000,
}

NUM_EPISODES = 5_000
LOG_EVERY = 100
SAVE_EVERY = 1_000


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


# ======================================================================
# Preprocessing
# ======================================================================
def preprocess(obs: np.ndarray) -> np.ndarray:
    """
    Transform the env's observation into what the network expects.

    The env with observation_mode="log2" already gives us log2 values
    in a (4, 4) float32 array. We just need to flatten (MLP) or
    one-hot encode (CNN) from the raw board.
    """
    if USE_CNN:
        raw = np.where(obs > 0, 2 ** obs, 0).astype(np.int64)
        return one_hot_encode(raw)  # shape: (16, 4, 4)
    else:
        return obs.flatten() / 17.0  # 17 ≈ log2(131072)


# ======================================================================
# Training
# ======================================================================
def train():
    # --- Environment ---
    env = Game2048Env(
        observation_mode="log2",
        reward_mode="score",
        invalid_move_penalty=-1.0,
    )

    # --- Networks ---
    if USE_CNN:
        q_net = DQN_CNN(in_channels=16, n_actions=4)
        target_net = DQN_CNN(in_channels=16, n_actions=4)
    else:
        q_net = DQN_MLP(input_dim=16, hidden=256, n_actions=4)
        target_net = DQN_MLP(input_dim=16, hidden=256, n_actions=4)

    # --- Agent ---
    agent = DQNAgent(
        q_network=q_net,
        target_network=target_net,
        device=DEVICE,
        **HYPERPARAMS,
    )

    print(f"Device: {DEVICE}")
    print(f"Network: {'CNN' if USE_CNN else 'MLP'}")
    print(f"Params: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"Training for {NUM_EPISODES} episodes...\n")

    # --- Logging ---
    reward_window = deque(maxlen=LOG_EVERY)
    max_tile_window = deque(maxlen=LOG_EVERY)
    loss_window = deque(maxlen=100)

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        state = preprocess(obs)
        episode_reward = 0
        done = False

        while not done:
            valid_actions = list(info["available_actions"])
            action = agent.select_action(state, valid_actions)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess(next_obs)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                loss_window.append(loss)

            state = next_state
            episode_reward += reward

        # --- Episode stats ---
        reward_window.append(episode_reward)
        max_tile_window.append(info["max_tile"])

        if episode % LOG_EVERY == 0:
            avg_reward = np.mean(reward_window)
            avg_loss = np.mean(loss_window) if loss_window else 0.0
            max_tile = max(max_tile_window)
            print(
                f"Ep {episode:>6d} | "
                f"Avg Reward: {avg_reward:>8.1f} | "
                f"Max Tile: {max_tile:>5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):,}"
            )

        if episode % SAVE_EVERY == 0:
            agent.save(f"checkpoint_ep{episode}.pt")

    agent.save("final_model.pt")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
