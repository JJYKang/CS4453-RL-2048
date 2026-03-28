"""
State preprocessing for 2048 boards.

The raw board is a 4x4 grid with values like 0, 2, 4, 8, 16, ...
Neural networks work better with normalized inputs, so we provide
two encoding options:

  - log2_preprocess:  Simple, works with DQN_MLP
  - one_hot_encode:   Richer representation, works with DQN_CNN
"""

import numpy as np


def log2_preprocess(board: np.ndarray, max_power: int = 16) -> np.ndarray:
    """
    Log2 normalize the board and flatten to a 1D vector.

    Each tile value v becomes log2(v) / max_power, with empty tiles = 0.
    Returns shape (16,) — ready for DQN_MLP.

    Example:
        [0, 2, 4, 8]  →  [0.0, 0.0625, 0.125, 0.1875, ...]
    """
    board = np.array(board, dtype=np.float32).flatten()
    # Avoid log2(0): replace 0 with 1 before taking log, then zero out
    safe = np.where(board > 0, board, 1)
    board = np.where(board > 0, np.log2(safe), 0)
    return board / max_power


def one_hot_encode(board: np.ndarray, n_channels: int = 16) -> np.ndarray:
    """
    One-hot encode the board into a (n_channels, 4, 4) tensor.

    Channel i is 1 where the tile value equals 2^i (channel 0 = empty).
    Returns shape (n_channels, 4, 4) — ready for DQN_CNN.

    Example:
        tile value 8 = 2^3  →  channel 3 is 1 at that position
    """
    board = np.array(board, dtype=np.int64).reshape(4, 4)
    encoded = np.zeros((n_channels, 4, 4), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val == 0:
                encoded[0, i, j] = 1.0
            else:
                # log2(val) gives the channel index (2=1, 4=2, 8=3, ...)
                channel = int(np.log2(val))
                if channel < n_channels:
                    encoded[channel, i, j] = 1.0

    return encoded
