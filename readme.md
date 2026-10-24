# RL-IF

[![Paper](https://img.shields.io/badge/Paper-arXiv-00bfff.svg)](https://arxiv.org/abs/2610.28883)
[![Conference](https://img.shields.io/badge/NeurIPS-2026-8a2be2.svg)](https://neurips.cc/virtual/2026/poster/152424)

This repository contains the official code for the NeurIPS 2026 paper [RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning](https://arxiv.org/abs/2610.28883).

RL-IF estimates how individual training instances influence a policy across many update steps (non-local influence) in online RL, and uses these estimates to filter training data. The code covers two settings: PPO training on standard RL benchmarks and RLHF fine-tuning for toxicity reduction.

## Repository Structure

This repository is organized as follows:

- [ppo_sb3/](ppo_sb3/): contains the RL-IF influence computation (`RLIF`, `RLIF-opt`) built on Stable-Baselines3, along with experiments in traditional RL environments from Gymnasium (e.g., LunarLander, BipedalWalker, FrozenLake, MuJoCo), using RL-IF to improve PPO training. See [ppo_sb3/README.md](ppo_sb3/README.md).
- [rlhf/](rlhf/): contains experiments on using RL-IF to improve the performance of RLHF in the task of LLM detoxification. See [rlhf/readme.md](rlhf/readme.md).

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{liu2026rlif,
  title     = {{RL-IF}: Tracking Non-local Training Data Influence for Online Reinforcement Learning},
  author    = {Liu, Shixuan and Tang, Cheng and Hu, Yuzheng and Wu, Fan and Zhao, Han and Ma, Jiaqi W.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
