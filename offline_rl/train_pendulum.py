import gym
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool
from action_wrappers import AlgoToEnvActionScalingWrapper

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

wandb.init(
    project='recurrent-ddpg-sac',
    entity='pomdpr',
    group='sac-pendulum-mdp',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

env = AlgoToEnvActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2)
buf = ReplayBuffer(capacity=60000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

batch_size = 64
num_episodes = 1000

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_updates = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)
        next_obs, reward, done, _ = env.step(action)
        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(Transition(obs, action, reward, next_obs, done))

        # ==================================================
        # update the parameters
        # ==================================================

        if buf.ready_for(batch_size):
            param.update_networks(buf.sample(batch_size))
            total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    wandb.log({'return': total_reward})

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Updates {total_updates:4.0f}')

env.close()