import gym
from gym.wrappers import Monitor
from params_pool import ParamsPool
from action_wrappers import ScalingActionWrapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

env_raw = gym.make('Pendulum-v0')
env = Monitor(
    ScalingActionWrapper(
        env_raw,
        scaling_factors=env_raw.action_space.high
    ),
    directory=f'results/trained_policies_video/{args.run_id}',
    force=True
)

param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)
param.load_actor(save_dir='results/trained_policies_pth', filename=f'{args.run_id}.pth')

obs = env.reset()

while True:
    next_obs, reward, done, _ = env.step(param.act(obs))
    if done: break
    obs = next_obs

env.close()