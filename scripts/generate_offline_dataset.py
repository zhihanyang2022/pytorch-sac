import numpy as np
import gym
import h5py
from tqdm import tqdm
from replay_buffer import Transition

env = gym.make('Pendulum-v0')

num_episodes = 1000
timeout = 200
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

observations = np.zeros((num_episodes * timeout, obs_dim))
actions = np.zeros((num_episodes * timeout, action_dim))
rewards = np.zeros((num_episodes * timeout, ))
next_observations = np.zeros((num_episodes * timeout, obs_dim))
dones = np.zeros((num_episodes * timeout, ))

pointer = 0
for e in tqdm(range(num_episodes)):
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        observations[pointer] = obs
        actions[pointer] = action
        rewards[pointer] = reward
        next_observations[pointer] = next_obs
        dones[pointer] = int(done)
        pointer += 1

        if done: break

with h5py.File('datasets/pendulum-random-v0.h5', 'w') as hf:
    hf.create_dataset("observations", data=observations)
    hf.create_dataset("actions", data=actions)
    hf.create_dataset("rewards", data=rewards)
    hf.create_dataset("next_observations", data=next_observations)
    hf.create_dataset("dones", data=dones)