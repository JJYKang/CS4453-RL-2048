# DQN for 2048

## Structure

```
├── train.py                 # Entry point — run this
└── dqn/
    ├── __init__.py          # Exports: DQNAgent, DQN_MLP, DQN_CNN, ReplayBuffer
    ├── agent.py             # DQN agent (epsilon-greedy, learning, target sync)
    ├── networks.py          # Q-network architectures (MLP + CNN)
    ├── replay_buffer.py     # Experience replay buffer
    └── preprocess.py        # Board → tensor (log2 or one-hot)
```

## Setup

```bash

path/to/CS4453-RL-2048

# 2. Install dependencies
pip install torch numpy gymnasium

# 3. Train
python train.py
```

## Switching to CNN

In `train.py`, set `USE_CNN = True`.

## Key Hyperparameters

| Param              | Default  | Notes                            |
|--------------------|----------|----------------------------------|
| lr                 | 5e-4     | Lower if training is unstable    |
| gamma              | 0.99     | Discount factor                  |
| batch_size         | 64       | 32 also works                    |
| target_update_freq | 1,000    | How often to sync target net     |
| eps_decay_steps    | 100,000  | Longer = more exploration        |
| buffer_capacity    | 100,000  | Bigger = more diverse samples    |

## Enhancements to Try

- **Double DQN**: Use q_net to *select* best action, target_net to *evaluate* it
- **Dueling DQN**: Split network into value stream + advantage stream
- **Prioritized replay**: Sample high TD-error transitions more often
- **Reward shaping**: Try `reward_mode="log_score"` or a custom reward_fn
