# CS4453-RL-2048


## Environment
High-level project entry point for RL development around the 2048 environment.

- Detailed environment API docs: [`src/rl2048/envs/README.md`](src/rl2048/envs/README.md)
- Primary environment implementation: `src/rl2048/envs/game_2048_env.py`


## Scripts
Repository entry points for project workflows, including environment checks and
future training/evaluation utilities.

- `scripts/random_rollout.py`: quick random-agent rollout against the environment.
- `scripts/train_dqn.py`: DQN training entrypoint driven by YAML config.
- Additional project scripts can be added under `scripts/` as training and
  evaluation work progresses.

## Configs
Configuration files for experiments and training jobs.

- `configs/dqn.yaml`: default DQN training and environment hyperparameters.
- Run training with:

```bash
make train-dqn
```

## Tests
Automated validation for environment and board-logic correctness.

- `tests/test_logic.py`: board logic unit tests.
- `tests/test_env.py`: Gym environment behavior tests.
- Smoke run:

```bash
make smoke
```

- Run tests:

```bash
make test
```

- Optional lint:

```bash
make lint
```
