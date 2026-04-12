"""
PPO Agent for 2048.

The key difference from DQN:
  - DQN asks "what's the value of each action?" and picks the best
  - PPO asks "what's the probability I should take each action?" and samples

The clipped surrogate objective is what makes PPO work:
  - It compares the new policy to the old policy
  - If the new policy is too different (ratio far from 1.0), it clips
  - This prevents catastrophically large updates
"""

import numpy as np
import torch
import torch.nn as nn
from rl2048.ppo.ppo_networks import PPOActorMLP, PPOCriticMLP
from rl2048.ppo.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        actor: PPOActorMLP,
        critic: PPOCriticMLP,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Separate optimizers (common in PPO)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            capacity=rollout_steps,
            state_shape=(16,),  # flattened board
            device=device,
        )

        self.steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, valid_actions: list[int] | None = None):
        """
        Sample action from current policy.
        Returns: (action, log_prob, value)
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_t, valid_actions)
            value = self.critic(state_t)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------
    def store(self, state, action, reward, done, log_prob, value, valid_actions):
        """Store a transition in the rollout buffer."""
        valid_mask = np.zeros(4, dtype=bool)
        for a in valid_actions:
            valid_mask[a] = True

        self.buffer.store(state, action, reward, done, log_prob, value, valid_mask)

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self, last_state: np.ndarray) -> dict:
        """
        Perform PPO update using collected rollout data.
        Call this after rollout_steps transitions have been collected.

        Returns dict of metrics for logging.
        """
        # Get value of last state for GAE bootstrap
        with torch.no_grad():
            last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state_t).item()

        # Compute advantages
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Track metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Multiple epochs over the same data
        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                valid_masks = batch["valid_masks"]

                # Normalize advantages (standard practice, reduces variance)
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # --- Actor (policy) loss ---
                new_log_probs, entropy = self.actor.evaluate(
                    states, actions, valid_masks
                )

                # Probability ratio: π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # --- Critic (value) loss ---
                values = self.critic(states)
                value_loss = nn.functional.mse_loss(values, returns)

                # --- Combined update ---
                actor_loss = policy_loss + self.entropy_coeff * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.buffer.reset()
        self.steps += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    @property
    def ready_to_update(self) -> bool:
        """Check if we've collected enough steps for an update."""
        return self.buffer.full

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "steps": self.steps,
            },
            path,
        )
        print(f"Saved PPO checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.steps = ckpt["steps"]
        print(f"Loaded PPO checkpoint from {path} (update {self.steps})")
