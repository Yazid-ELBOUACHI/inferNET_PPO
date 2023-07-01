import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from rlenv.envs.wall.core import AccentaEnv
from stable_baselines3 import PPO

# Define the InferNet model architecture
infernet_model = keras.Sequential([
    layers.TimeDistributed(layers.Dense(256, activation='relu'), input_shape=(None, 4)),
    layers.TimeDistributed(layers.Dense(256, activation='relu')),
    layers.TimeDistributed(layers.Dense(256, activation='relu')),
    layers.TimeDistributed(layers.Dense(1))  # Output a scalar reward
])

# Define the PPO agent model architecture
ppo_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Output action probabilities for 2 actions
])

# Initialize InferNet buffer D
D = []

# Hyperparameters
K = 100  # Number of pretraining episodes
M = 200  # Number of training episodes
batch_size = 32
gamma = 0.99  # Discount factor

# Pretrain InferNet
for episode in range(K):
    # Play an episode randomly and collect the data
    env = AccentaEnv()
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False
    while not done:
        # Select a random action
        action = env.action_space.sample()

        # Interact with the environment
        next_state, reward, done, _ = env.step(action)

        # Store the data
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    # Calculate delayed reward Rdel
    Rdel = sum(rewards)

    # Add episode data to buffer D
    D.append((states, actions, rewards, Rdel))

    # Sample mini-batch of episodes B from D
    batch_indices = np.random.choice(len(D), batch_size, replace=True)
    batch = [D[idx] for idx in batch_indices]

    # Train InferNet on the mini-batch
    states_batch = np.concatenate([data[0] for data in batch])
    actions_batch = np.concatenate([data[1] for data in batch])
    rewards_batch = np.concatenate([data[2] for data in batch])
    Rdel_batch = np.array([data[3] for data in batch])
    with tf.GradientTape() as tape:
        infer_rewards = infernet_model(states_batch, training=True)
        infer_rewards = tf.reshape(infer_rewards, [-1])  # Reshape to remove sequence dimension
        loss = tf.reduce_mean(tf.square(Rdel_batch - infer_rewards))
    gradients = tape.gradient(loss, infernet_model.trainable_variables)
    infernet_model.optimizer.apply_gradients(zip(gradients, infernet_model.trainable_variables))

# Train PPO RL agent using InferNet
model = PPO(ppo_model, verbose=1)  # Create PPO agent with ppo_model as the policy network
for episode in range(M):
    tmp = []

    # Collect episode data
    env = AccentaEnv()
    state = env.reset()
    done = False
    while not done:
        # Get state s from the environment
        state = env.get_state()
        # Select action a from policy π
        action_probs = ppo_model.predict(np.expand_dims(state, axis=0))
        action_probs = tf.squeeze(action_probs)
        action = np.random.choice(len(action_probs), p=action_probs)

        # Interact with the environment and get next state s', reward r
        next_state, reward, done, _ = env.step(action)

        # Add data to tmp sequence
        tmp.append((state, action, reward, next_state))

        # Train PPO RL agent
        model.learn(total_timesteps=1000000)

        state = next_state

    # Reshape tmp sequence for TimeDistributed input
    states_tmp = np.expand_dims(np.array([data[0] for data in tmp]), axis=0)

    # Use InferNet to infer rewards for the steps in tmp
    infer_rewards = infernet_model(states_tmp, training=False)
    infer_rewards = tf.reshape(infer_rewards, [-1])  # Reshape to remove sequence dimension

    # Replace rewards in tmp with InferNet rewards
    for i in range(len(tmp)):
        tmp[i] = (tmp[i][0], tmp[i][1], infer_rewards[i].numpy(), tmp[i][3])

    # Add tmp sequence to buffer D
    D.extend(tmp)


# Store data in tmp to train the PPO RL agent later on
for data in tmp:
    D.append(data)

# Train the PPO RL agent using the updated buffer D
if len(D) >= batch_size:
    # Sample mini-batch of episodes B from D
    batch_indices = np.random.choice(len(D), batch_size, replace=False)
    batch = [D[idx] for idx in batch_indices]

    # Prepare the mini-batch data
    states_batch = np.array([data[0] for data in batch])
    actions_batch = np.array([data[1] for data in batch])
    rewards_batch = np.array([data[2] for data in batch])
    next_states_batch = np.array([data[3] for data in batch])

    # Train the PPO RL agent on the mini-batch
    model.learn(total_timesteps=1000000, states_batch=states_batch, actions_batch=actions_batch, rewards_batch=rewards_batch, next_states_batch=next_states_batch)

# Evaluate the trained model
score = AccentaEnv.eval(model)
print(score)