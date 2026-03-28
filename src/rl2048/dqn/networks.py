"""
Q-Network architectures for 2048 DQN.

Two options:
  - DQN_MLP:  Simple fully-connected network (good starting point)
  - DQN_CNN:  Convolutional network treating board as 4x4 image with
              one-hot encoded tile channels (better spatial reasoning)

Both take a preprocessed board state and output 4 Q-values
(one per action: up, down, left, right).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Option A: Fully-Connected / MLP
# ---------------------------------------------------------------------------
class DQN_MLP(nn.Module):
    """
    Input:  flattened board of shape (batch, 16)
            (use log2 preprocessing — see preprocess.py)
    Output: Q-values of shape (batch, 4)
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
        return self.net(x)


# ---------------------------------------------------------------------------
# Option B: Convolutional Network
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    Multi-kernel conv block: runs 1x1, 2x2, 3x3, 4x4 convolutions
    in parallel and concatenates them. This lets the network see
    patterns at different spatial scales simultaneously.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        d = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, d, kernel_size=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, d, kernel_size=2, padding="same")
        self.conv3 = nn.Conv2d(in_channels, d, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(in_channels, d, kernel_size=4, padding="same")

    def forward(self, x):
        return torch.cat([
            F.relu(self.conv1(x)),
            F.relu(self.conv2(x)),
            F.relu(self.conv3(x)),
            F.relu(self.conv4(x)),
        ], dim=1)


class DQN_CNN(nn.Module):
    """
    Input:  one-hot encoded board of shape (batch, n_tile_types, 4, 4)
            where n_tile_types = 16 (tiles: 0, 2, 4, 8, ..., 2^15=32768)
            (see preprocess.py for one_hot_encode)
    Output: Q-values of shape (batch, 4)
    """

    def __init__(self, in_channels: int = 16, n_actions: int = 4):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 128)
        self.conv2 = ConvBlock(128, 128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
