"""
Rollout Buffer for PPO.

Unlike DQN's replay buffer (random sampling from history), PPO collects
a fixed number of steps, uses ALL of them for updates, then throws them
away. This is "on-policy" learning — you only learn from data your
current policy generated.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, capacity: int, state_shape: tuple, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.pos = 0
        self.full = False

        # Pre-allocate arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.valid_masks = np.zeros((capacity, 4), dtype=bool)  # 4 actions

        # Computed after rollout is complete
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def store(self, state, action, reward, done, log_prob, value, valid_mask):
        """Store one transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.valid_masks[self.pos] = valid_mask
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """
        Compute Generalized Advantage Estimation (GAE).

        This is the key insight of PPO — instead of just using raw rewards,
        GAE blends short-term and long-term advantage estimates for lower
        variance while keeping some bias. The lambda parameter controls
        this tradeoff (1.0 = Monte Carlo, 0.0 = TD(0)).
        """
        n = self.pos
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD error: r + γV(s') - V(s)
            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )

            # GAE: weighted sum of TD errors
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """
        Yield random minibatches from the rollout.
        PPO typically does multiple passes (epochs) over the same data.
        """
        n = self.pos
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                continue  # drop incomplete batch

            idx = indices[start:end]
            yield {
                "states": torch.FloatTensor(self.states[idx]).to(self.device),
                "actions": torch.LongTensor(self.actions[idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(self.log_probs[idx]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[idx]).to(self.device),
                "returns": torch.FloatTensor(self.returns[idx]).to(self.device),
                "valid_masks": torch.BoolTensor(self.valid_masks[idx]).to(self.device),
            }

    def reset(self):
        """Clear buffer for next rollout."""
        self.pos = 0
        self.full = False
