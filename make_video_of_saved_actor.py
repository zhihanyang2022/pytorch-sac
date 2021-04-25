import gym
from gym.wrappers import Monitor
from params_pool import ParamsPool
from action_wrappers import ScalingActionWrapper
import argparse

def make_video_of_saved_actor(save_dir: str, filename: str) -> None:

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
    param.load_actor(save_dir=save_dir, filename=filename)

    obs = env.reset()

    while True:
        next_obs, reward, done, _ = env.step(param.act(obs))
        if done: break
        obs = next_obs

    env.close()