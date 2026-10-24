# Non-Local RL Attribution: RL-IF

Code for training PPO agents with influence-function-based data selection.

## Methods

| Method | Description |
|--------|-------------|
| **RLIF-opt** | Optimization approximation of RLIF |
| **RLIF** | RL Influence |
| **TracIn** | TracIn influence estimation |
| **Random** | Uniform random data selection baseline |
| **Baseline** | Standard PPO with no data selection |

## Installation

Requires Python 3.10+, PyTorch, and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

```bash
pip install stable-baselines3 gymnasium tqdm
```

## Usage

```bash
python run_experiments.py <config_file> [--seeds 42] [--device cpu]
```

Each experiment config is a YAML file that composes four sub-configs:

```yaml
training_config:
  env_name: Ant-v4
  experiment_name: rlif_opt_h5_kl5.0
  data_selection_method: global   # global | local | none
  n_envs: 8
  total_rollouts: 1000
  rollout_begin: 0
  rollout_end: 1000

ppo_config: configs/trpo/ant.yaml
ppo_override:
  kl_coef: 5.0

attribution_config: configs/attribution/rlif_opt_x10.0_h5.yaml
attribution_override:
  step_size: 5        # horizon H

eval_config: configs/eval/parallel_eval.yaml
```