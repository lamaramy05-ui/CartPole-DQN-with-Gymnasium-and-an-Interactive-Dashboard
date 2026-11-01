# CartPole DQN with Gymnasium and an Interactive Dashboard

## Overview

This project trains a Deep Q-Network (DQN) agent to balance a pole on a moving cart in the **CartPole-v1** environment from Gymnasium.

The DQN is implemented directly in PyTorch rather than using a pre-built reinforcement-learning agent. The notebook covers environment setup, training, evaluation, a reward plot, model saving, and an interactive dashboard for replaying the trained agent's decisions.

**Main goal:** keep the pole balanced for as many time steps as possible. The environment gives the agent a reward of **+1 for each time step** it stays alive, up to a maximum of **500 steps per episode**.

## Features

- Creates and inspects the `CartPole-v1` environment.
- Implements a Q-network, replay buffer, and target network.
- Uses epsilon-greedy exploration during training.
- Trains the agent with a DQN learning target and Huber (smooth L1) loss.
- Plots episode rewards and a 20-episode moving average.
- Evaluates the trained policy over 20 separate episodes without exploration.
- Saves the trained policy weights to `cartpole_dqn.pth`.
- Includes an interactive dashboard with **Play**, **Pause**, **Reset**, and speed controls to replay an episode and inspect the agent's state, action, and reward.

## Environment and actions

At each time step, the agent receives four measurements:

1. Cart position
2. Cart velocity
3. Pole angle
4. Pole angular velocity

It chooses one of two actions:

- `0` — push the cart left
- `1` — push the cart right

The agent does not control the pole directly. It learns which cart movement is likely to help keep the pole upright.

## How DQN works in this notebook

The notebook uses these main components:

- **Q-network:** a neural network with two hidden layers of 128 units each. It estimates the expected return for each available action.
- **Replay buffer:** stores previous transitions and samples random mini-batches for learning.
- **Target network:** provides more stable learning targets and is synchronized with the policy network periodically.
- **Epsilon-greedy exploration:** the agent sometimes chooses a random action early in training and increasingly relies on its learned Q-values.
- **Discount factor (`GAMMA`):** `0.99`, so future rewards are included in the learning target.
- **Batch size:** `64`.
- **Optimizer:** Adam with a learning rate of `0.001`.
- **Training limit:** up to `600` episodes, with a maximum of `500` steps per episode.
- **Solve criterion:** a mean reward of at least `475` over the most recent `100` episodes. Reaching this criterion stops training early.

## Requirements

The notebook installs these packages in its first setup cell:

- Python (a notebook environment such as Google Colab is recommended)
- Gymnasium with classic-control support
- PyTorch
- NumPy
- Matplotlib
- Pillow
- ipywidgets

The notebook's installation command is:

```bash
pip install "gymnasium[classic-control]" torch matplotlib numpy pillow ipywidgets
```

## How to run

### Recommended: Google Colab

1. Upload `CartPole_DQN_Gymnasium_Improved_UI.ipynb` to Google Colab, or open it from your Google Drive.
2. Select a runtime. A GPU is optional; the notebook automatically uses CUDA when available and otherwise uses the CPU.
3. Run the notebook cells **from top to bottom**. The first code cell installs the required packages.
4. Allow the training cell to finish. It can run for up to 600 episodes unless the solve criterion is reached earlier.
5. Review the reward plot and evaluation summary.
6. Use the interactive dashboard controls to replay the recorded episode and explore the agent's decisions.

The notebook sets the random seed to `42` for Python, NumPy, and PyTorch. Environment resets during training and evaluation use their own fixed seed sequences. Reinforcement-learning results can still vary with runtime, software versions, and execution details.

### Saved model

The notebook saves the model checkpoint to:

```text
/content/cartpole_dqn.pth
```

This path is intended for Google Colab. Download the file from the Colab runtime if you want to keep it after the session ends. The checkpoint contains the policy network's weights and the state/action dimensions. To load it in a fresh runtime, first define the same `QNetwork` class, then use the optional loading example at the end of the notebook.

## Results from the notebook's recorded run

The uploaded notebook contains the following evaluation output:

| Metric | Recorded value |
|---|---:|
| Training episodes completed | 600 |
| Best training episode reward | 227 |
| Average reward over the final 100 training episodes | 46.08 |
| Evaluation episodes | 20 |
| Mean evaluation reward | 160.75 |
| Evaluation reward standard deviation | 60.41 |
| Lowest evaluation reward | 89 |
| Highest evaluation reward | 318 |
| Recorded dashboard replay | 91 steps |

These are the values recorded in the notebook, not a guarantee that a new run will produce the same results.

### What the results mean

The agent learned some useful behavior, but **the recorded run did not solve CartPole**. Its final-100 training average of `46.08` and mean evaluation reward of `160.75` are below the notebook's solve target of `475`. The evaluation scores also vary noticeably between episodes.

The reward-versus-episodes plot is useful for seeing how learning changes over time. A score near `500` means the agent kept the pole balanced for almost the full episode. If performance remains low, the notebook suggests rerunning training or increasing `TRAIN_EPISODES`, then comparing the reward plot and evaluation results. Any change should be tested with a fresh evaluation rather than judged from a single episode.

## Notebook structure

| Section | Purpose |
|---|---|
| 1. Install and import libraries | Set up packages, seeds, and device |
| 2. Inspect the environment | Check observation and action spaces |
| 3. Define DQN components | Build the Q-network, replay buffer, and training step |
| 4. Train the agent | Collect experiences and update the networks |
| 5. Plot rewards | Visualize training rewards and moving average |
| 6. Save the model | Write the checkpoint to disk |
| 7. Evaluate the agent | Measure performance without exploration |
| 8. Interactive dashboard | Replay an episode and inspect decisions |
| 9. Results and conclusion | Interpret the training and evaluation |
| 10. Optional model loading | Restore a checkpoint in a fresh runtime |

## Limitations

- The current recorded run does not meet the defined solve criterion.
- The dashboard replays a sequence collected from the policy; it is not a live retraining interface.
- The model checkpoint is saved to a Colab-specific path by default.
- Results may vary between runs. Evaluation over multiple episodes is more informative than a single successful or unsuccessful episode.

## Possible next steps

1. Increase the training budget and compare the moving-average reward.
2. Check learning behavior and hyperparameters if rewards remain low.
3. Evaluate every saved or improved policy over multiple episodes.
4. Keep the evaluation procedure separate from training so evaluation episodes do not update the network.

## Project file

- `CartPole_DQN_Gymnasium_Improved_UI.ipynb` — complete notebook containing the implementation, training, evaluation, plots, checkpoint saving, and interactive dashboard.
