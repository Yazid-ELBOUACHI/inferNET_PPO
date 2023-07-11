import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from stable_baselines3 import PPO
from rlenv.envs.wall.core import AccentaEnv

# Define the InferNet model architecture
infernet_model = keras.Sequential([
    layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.2), input_shape=(None, 14)),
    layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.2)),
    layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.2)),
    layers.Dense(1)
])

# Define the PPO agent model architecture
ppo_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(None,14)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)
])

# Initialize InferNet buffer D
D = []

# Hyperparameters
K = 100000 # Number of pretraining episodes
M = 100000
batch_size = 128
optimizer = tf.keras.optimizers.Adam()  # Initialize the Adam optimizer
env = AccentaEnv()

# Pretrain InferNet
for episode in range(K):
    print("=================================================================================================")
    print("episode number ", episode)
    # Play an episode randomly and collect the data
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
    optimizer = tf.keras.optimizers.Adam()  # Initialize the Adam optimizer

    with tf.GradientTape() as tape:
        states_batch = np.concatenate([data[0] for data in batch])
        actions_batch = np.concatenate([data[1] for data in batch])
        rewards_batch = np.concatenate([data[2] for data in batch])
        Rdel_batch = np.array([data[3] for data in batch])
        Rdel_batch = tf.cast(Rdel_batch, dtype=tf.float32)
        Rdel_batch = tf.expand_dims(Rdel_batch, axis=-1)
        states_with_actions = np.concatenate([states_batch, actions_batch], axis=-1)
        infer_rewards = infernet_model(states_with_actions, training=True)
        infer_sum_rewards = tf.reduce_sum(infer_rewards, axis=1)
        infer_sum_rewards = tf.cast(infer_sum_rewards, dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(Rdel_batch - infer_sum_rewards))
        print("loss =", loss)

    gradients = tape.gradient(loss, infernet_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, infernet_model.trainable_variables))

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

        state = next_state

    # Train InferNet on the collected episode data
    episode_states = np.array([data[0] for data in tmp])
    episode_actions = np.array([data[1] for data in tmp])
    episode_states_with_actions = np.concatenate([episode_states, episode_actions], axis=-1)

    with tf.GradientTape() as tape:
        episode_infer_rewards = infernet_model(episode_states_with_actions, training=True)
        episode_infer_sum_rewards = tf.reduce_sum(episode_infer_rewards, axis=1)
        episode_infer_sum_rewards = tf.cast(episode_infer_sum_rewards, dtype=tf.float32)

        episode_loss = -tf.reduce_mean(episode_infer_sum_rewards)  # Maximize the inferred rewards

    episode_gradients = tape.gradient(episode_loss, infernet_model.trainable_variables)
    optimizer.apply_gradients(zip(episode_gradients, infernet_model.trainable_variables))

    # Infer immediate rewards using InferNet and store them in the tmp buffer
    tmp_infer_rewards = infernet_model(episode_states_with_actions, training=False)
    tmp = [(data[0], data[1], tmp_infer_rewards[i].numpy(), data[3]) for i, data in enumerate(tmp)]

    # Add tmp sequence to buffer D
    D.extend(tmp)

    # Train PPO RL agent
    model.learn(total_timesteps=M, reset_num_timesteps=False, log_interval=10)

    # Reshape tmp sequence
    states_tmp = np.expand_dims(np.array([data[0] for data in tmp]), axis=0)
    actions_tmp = np.array([data[1] for data in tmp])
    states_with_actions_tmp = np.concatenate([states_tmp, actions_tmp], axis=-1)

    # Use InferNet to infer rewards for the steps in tmp
    infer_rewards = infernet_model(states_with_actions_tmp, training=False)

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
    states_with_actions_batch = np.concatenate([states_batch, actions_batch], axis=-1)

    # Train the PPO RL agent on the mini-batch
    model.learn(total_timesteps=1000000, states_batch=states_with_actions_batch, rewards_batch=rewards_batch, next_states_batch=next_states_batch)

# Evaluate the trained model
score = AccentaEnv.eval(model)
print(score)