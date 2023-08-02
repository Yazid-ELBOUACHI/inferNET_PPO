# InferNet PPO RL Agent

This repository contains code for training a Reinforcement Learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm combined with an InferNet model. The InferNet model is used to infer the rewards for an environment, which are then used to update the policy of the PPO agent.

## Dependencies
- TensorFlow 2.x
- Keras (from TensorFlow)
- NumPy
- Stable Baselines3

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Yazid-ELBOUAVHI/inferNET_PPO.git
```

2. Install the required dependencies:

```bash
pip install tensorflow keras numpy stable_baselines3
```

## How to Use

The main script in this repository is `infernet_ppo.py`. It consists of two parts: pretraining the InferNet model and training the PPO agent using the pre-trained InferNet.

### 1. Pretraining the InferNet model

To pretrain the InferNet model, run the following command:

```bash
python infernet_ppo.py --pretrain
```

The pretraining process involves playing random episodes in the environment and storing the state-action-reward sequences in a buffer `D`. The InferNet model is then trained on mini-batches sampled from `D` to approximate the delayed rewards `Rdel`.

### 2. Training the PPO agent using InferNet

To train the PPO agent using the pretrained InferNet model, run the following command:

```bash
python infernet_ppo.py --train
```

This will train the PPO agent using the `PPO` algorithm from Stable Baselines3 with the policy network represented by `ppo_model`. During training, the InferNet model is also updated using the rewards inferred from the PPO agent's interactions with the environment.

## Environment

The RL environment used in this code is `AccentaEnv` from the `rlenv` package. The environment provides an API to interact with the simulation and get state observations, action space, and rewards.

## Hyperparameters

The following hyperparameters are used in the code:

- `K`: Number of pretraining episodes
- `M`: Number of PPO training episodes using InferNet
- `batch_size`: Batch size for training InferNet
- `optimizer`: Adam optimizer used for training InferNet
- `ppo_model`: The policy network used for the PPO agent
- `infernet_model`: The InferNet model used for approximating delayed rewards

## Evaluation

To evaluate the trained PPO RL agent, you can use the following command:

```bash
python infernet_ppo.py --evaluate
```

This will evaluate the trained agent on the `AccentaEnv` and print the final score.

## Note

This code is provided as an example and may require modification to work with your specific RL problem and environment. Feel free to customize the architecture, hyperparameters, and training loop to suit your needs.

If you find this code useful or have any questions, please feel free to reach out.

Happy RL training!

