import gym
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool
from action_wrappers import AlgoToEnvActionScalingWrapper

import wandb
import argparse

# =================================================================================
# arguments

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
parser.add_argument('--offline_rl', action='store_true')  # default is false
parser.add_argument('--cql', action='store_true')  # default is false
parser.add_argument('--sil', action='store_true')
args = parser.parse_args()

if args.offline_rl is False:
    assert args.cql is False, "You can't use CQL when doing online learning."

if args.cql is False:
    assert args.sil is False, "You can't use the SIL idea without CQL."

# =================================================================================
# logging

group_name_base = 'pendulum-sac'

if args.offline_rl:
    if args.cql:
        if args.sil:
            group_name_postfix = 'offline-cql-sil'
        else:
            group_name_postfix = 'offline-cql'
    else:
        group_name_postfix = 'offline'
else:
    group_name_postfix = 'online'

wandb.init(
    project='offline-rl',
    entity='yangz2',
    group=f'{group_name_base}-{group_name_postfix}',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

# =================================================================================

env = AlgoToEnvActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2)
buf = ReplayBuffer(capacity=60000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

if args.offline_rl:
    pass
    # TODO: load dataset and push all transitions into the buffer
    # TODO: increase the size of replay buffer if necessary

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

        if args.offline_rl is False:
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