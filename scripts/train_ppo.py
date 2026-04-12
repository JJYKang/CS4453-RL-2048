"""
PPO Training loop for 2048.

The key structural difference from DQN training:

  DQN:  play one step → store → sample random batch → update
  PPO:  play N steps → compute advantages → update multiple epochs → discard data

PPO collects a "rollout" of fixed length, then does several passes
over that rollout to update the policy. Then it throws away the data
and collects a fresh rollout. This is on-policy learning.
"""

from collections import deque

import numpy as np
import torch
from rl2048.envs.game_2048_env import Game2048Env
from rl2048.ppo.ppo_agent import PPOAgent
from rl2048.ppo.ppo_networks import PPOActorMLP, PPOCriticMLP

# ======================================================================
# Config
# ======================================================================
ROLLOUT_STEPS = 2048  # Steps per rollout before updating
TOTAL_TIMESTEPS = 500_000  # Total env steps to train
LOG_EVERY = 10  # Log every N rollout updates
SAVE_EVERY = 50  # Checkpoint every N updates

PPO_PARAMS = {
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,  # The core PPO hyperparameter
    "entropy_coeff": 0.01,  # Higher = more exploration
    "value_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,  # Passes over each rollout
    "batch_size": 64,
    "rollout_steps": ROLLOUT_STEPS,
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def preprocess(obs: np.ndarray) -> np.ndarray:
    """Flatten log2 board and normalize."""
    return obs.flatten() / 17.0


def train():
    env = Game2048Env(
        observation_mode="log2",
        reward_mode="score",
        invalid_move_penalty=-1.0,
    )

    actor = PPOActorMLP(input_dim=16, hidden=256, n_actions=4)
    critic = PPOCriticMLP(input_dim=16, hidden=256)

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        device=DEVICE,
        **PPO_PARAMS,
    )

    print(f"Device: {DEVICE}")
    print(f"Actor params:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"Rollout steps: {ROLLOUT_STEPS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Updates planned: ~{TOTAL_TIMESTEPS // ROLLOUT_STEPS}\n")
    # --- Ploting ---
    import csv
    from pathlib import Path

    log_dir = Path("runs/ppo_checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "episode_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        [
            "update",
            "timesteps",
            "avg_reward",
            "max_tile",
            "policy_loss",
            "value_loss",
            "entropy",
        ]
    )
    # --- Logging ---
    episode_rewards = deque(maxlen=100)
    episode_max_tiles = deque(maxlen=100)
    total_steps = 0
    update_count = 0

    # Start first episode
    obs, info = env.reset()
    state = preprocess(obs)
    episode_reward = 0.0

    while total_steps < TOTAL_TIMESTEPS:
        valid_actions = list(info["available_actions"])

        # Select action from policy
        action, log_prob, value = agent.select_action(state, valid_actions)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_obs)
        done = terminated or truncated

        # Store in rollout buffer
        agent.store(state, action, reward, done, log_prob, value, valid_actions)
        total_steps += 1
        episode_reward += reward

        # Episode ended
        if done:
            episode_rewards.append(episode_reward)
            episode_max_tiles.append(info["max_tile"])

            obs, info = env.reset()
            state = preprocess(obs)
            episode_reward = 0.0
        else:
            state = next_state

        # --- PPO Update ---
        if agent.ready_to_update:
            metrics = agent.update(state)
            update_count += 1

            if update_count % LOG_EVERY == 0 and len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                max_tile = max(episode_max_tiles) if episode_max_tiles else 0
                print(
                    f"Update {update_count:>5d} | "
                    f"Steps: {total_steps:>8,} | "
                    f"Avg Reward: {avg_reward:>8.1f} | "
                    f"Max Tile: {max_tile:>5d} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f}"
                )
                csv_writer.writerow(
                    [
                        update_count,
                        total_steps,
                        f"{avg_reward:.1f}",
                        max_tile,
                        f"{metrics['policy_loss']:.4f}",
                        f"{metrics['value_loss']:.4f}",
                        f"{metrics['entropy']:.4f}",
                    ]
                )
                log_file.flush()

            if update_count % SAVE_EVERY == 0:
                agent.save(f"ppo_checkpoint_u{update_count}.pt")

    agent.save("ppo_final.pt")
    log_file.close()
    env.close()
    print("\nPPO training complete!")


if __name__ == "__main__":
    train()
