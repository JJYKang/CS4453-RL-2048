"""
PPO Networks for 2048.

Actor:  takes state → outputs action probabilities (policy)
Critic: takes state → outputs single value estimate V(s)

Can share a backbone or be fully separate. We keep them
separate here for clarity — easier to debug and tune independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOActorMLP(nn.Module):
    """
    Policy network: state → action probabilities.
    Input:  (batch, 16) flattened log2 board
    Output: (batch, 4) action probabilities (softmax)
    """

    def __init__(self, input_dim: int = 16, hidden: int = 256, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits  # raw logits, softmax applied in get_action/evaluate

    def get_action(self, state, valid_actions=None):
        """
        Sample an action from the policy.
        Returns: (action, log_prob)
        """
        logits = self.forward(state)

        # Mask invalid actions
        if valid_actions is not None:
            mask = torch.full_like(logits, float("-inf"))
            for a in valid_actions:
                mask[..., a] = 0.0
            logits = logits + mask

        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, states, actions, valid_action_masks=None):
        """
        Evaluate actions taken — used during PPO update.
        Returns: (log_probs, entropy)
        """
        logits = self.forward(states)

        if valid_action_masks is not None:
            # mask shape: (batch, 4), True = valid
            logits = logits.masked_fill(~valid_action_masks, float("-inf"))

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class PPOCriticMLP(nn.Module):
    """
    Value network: state → V(s) scalar estimate.
    Input:  (batch, 16) flattened log2 board
    Output: (batch,) value estimate
    """

    def __init__(self, input_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
