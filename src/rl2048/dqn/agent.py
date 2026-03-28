"""
DQN Agent for 2048.

Handles:
  - Epsilon-greedy action selection (with invalid move masking)
  - Storing transitions in the replay buffer
  - Sampling minibatches and performing gradient updates
  - Syncing the target network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule (linear decay)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # Networks
        self.q_net = q_network.to(self.device)
        self.target_net = target_network.to(self.device)
        self.sync_target()  # initialize target with same weights
        self.target_net.eval()

        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — less sensitive to outliers

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # Step counter (for epsilon decay + target sync)
        self.steps = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Current epsilon value (linearly decayed)."""
        progress = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, valid_actions: list[int] | None = None) -> int:
        """
        Epsilon-greedy action selection with optional invalid-move masking.

        Args:
            state: Preprocessed board state (numpy array).
            valid_actions: List of valid action indices (e.g. [0, 1, 3]).
                           If None, all 4 actions are considered valid.
                           *** Your teammate's env should provide this. ***

        Returns:
            Chosen action index (0=up, 1=down, 2=left, 3=right — or
            whatever mapping your env uses).
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]

        # Explore
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Exploit
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # Mask invalid actions by setting their Q-values to -inf
        masked_q = np.full(4, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """
        Sample a minibatch and do one gradient step.
        Returns the loss value, or None if buffer isn't full enough yet.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # --- Sample ---
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # --- Current Q-values ---
        # q_net(states) → (batch, 4), gather picks the Q-value for the action taken
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # --- Target Q-values ---
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # --- Gradient step ---
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # --- Bookkeeping ---
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------
    def sync_target(self):
        """Copy q_net weights → target_net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        print(f"Loaded checkpoint from {path} (step {self.steps})")
