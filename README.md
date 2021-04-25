# Soft actor-critic on Pendulum-v0

*Special thanks to Joshua Achiam for creating OpenAI Spinning Up, which was the primary resource that enabled me to code up this implementation. In particular, his organized [pseudo-code](https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode) and minimalistic PyTorch [implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) were most helpful to me. Please check them out.*

Training returns             |  Expert policy
:-------------------------:|:-------------------------:
![](/results/training_returns_svg/Pendulum-v0.svg)  |  ![](/results/trained_policies_videos/5/openaigym.video.0.43160.video000000.mp4)

### What is SAC?

SAC is a deep reinforcement learning algorithm for continuous control in discrete/continuous state spaces. It has a similar structure to "Q-learning" algorithms (those that utilize some form of Bellman update instead of the policy-gradient update) such as DDPG and TD3. It is inspired by the policy iteration algorithm under the maximum entropy RL framework. 

### Features of this implementation

- Hyper-parameters
  - Learning rates, discount factor and polyak multiplier were borrowed from Joshua's implementation.
  - No automatic entropy tuning, just like Joshua's implementation.
  - Update rate: one gradient step per environment step
- Style
  - Joshua's implementation 
- Quality
  - TODO: shown to work on a variety of tasks including classical control task, and pybullet, and mujoco
  - Compared with Joshua's implementation

### How to get it running on your computer?

1. Create a conda environment and install dependencies
2. Log into your wandb account in command line using `wandb login`
3. Change the logging code in `train_pendulum.py` to use your account name and project name instead of mine
4. In command line, type `python train_pendulum.py --run_id=<some integer>` to train a single run.
5. Visualize the training curve in wandb conveniently.
6. After training has finished:
    - a trained actor will be saved automatically to `results/trained_policies_pth`;
    - a video of the expert policy will be saved automatically to `results/trained_policies_video`.
7. Repeat step 4 to 6 until you have enough runs.
8. (Optional) To export training returns from wandb, use `results/download_from_wandb.py`.
9. (Optional) To plot the exported training returns, use `results/plot_training_returns.py`.

### Lessons learned

- TODO: Do not use nn.softplus; show the gradient details
- TODO: SAC might be particularly sensitive to hyperparameters; show learning curves
