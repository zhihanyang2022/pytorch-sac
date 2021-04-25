# Soft actor-critic in PyTorch 🚀

*Special thanks to Joshua Achiam for creating OpenAI Spinning Up, which was the primary resource that enabled me to code up this implementation. In particular, his organized [pseudo-code](https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode) and minimalistic PyTorch [implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) were most helpful to me. Please check them out.*

### Table of content

### What is SAC?

SAC is a deep reinforcement learning algorithm for continuous control in discrete/continuous state spaces. It has a similar structure to "Q-learning" algorithms (those that utilize some form of Bellman update instead of the policy-gradient update) such as DDPG and TD3. It is inspired by the policy iteration algorithm under the maximum entropy RL framework. 

### Features of this implementation

- Hyperparameters
  - Learning rates, discount factor and polyak multiplier were borrowed from Joshua's implementation.
  - No automatic entropy tuning, just like Joshua's implementation.
- Style
  - Joshua's implementation 
- Quality
  - TODO: shown to work on a variety of tasks including classical control task, and pybullet, and mujoco
  - Compared with Joshua's implementation

### Learning curves

<<<<<<< HEAD
<img src="results/training_returns_svg/Pendulum-v0.svg">
=======
- TODO: pendulum-v0, acrobot-v0 (I modified it to have continuous actions), cartpole-v0 (modified to have continuous actions)

### Sensitivity to hyper-parameters
>>>>>>> 85758dcbc767da7880fc8fd4d26f5d91a3a14abd

### Lessons learned

- TODO: Do not use nn.softplus; show the gradient details
- TODO: SAC might be particularly sensitive to hyperparameters; show learning curves

### How to get it running on your computer?

TODO: requirements.txt
