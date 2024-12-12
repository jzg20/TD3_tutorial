# TD3 Training on InvertedPendulum-v5

This repository contains a Python script to train an agent using Twin Delayed Deep Deterministic Policy Gradient (TD3), the reinforcement learning algorithm. The agent has been trained on InvertedPendulum-v5 environment from the OpenAI Gymnasium framework. Stablebaseline3, which is a set of RL implementations using PyTorch, has been used in training.

**Video Saving**
Saves videos of the 3D inverted pendulum for episodes with the best rewards.

**Visualisations:**
- Learning Curve (Reward vs Time Step)
- Action Distribution
- Best Rewards Over Time
- Smoothed Rewards
- Reward vs Episode

**Customizable parameters**
- num_runs - number of runs
- gym.make('InvertedPendulum-v5') - environment
- save_path - directory where results are saved
