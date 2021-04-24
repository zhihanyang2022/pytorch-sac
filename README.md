# Soft actor-critic

*Special thanks to Joshua Achiam for creating OpenAI Spinning Up, which was the primary resource that enabled me to code up this implementation. In particular, his organized [pseudo-code](https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode) and minimalistic PyTorch [implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) were most helpful to me. Please check them out.*

### What is SAC?

SAC is a deep reinforcement learning algorithm for continuous control in discrete/continuous state spaces. It has a similar structure to "Q-learning" algorithms (those that utilize some form of Bellman update instead of the policy-gradient update) such as DDPG and TD3. It is inspired by the policy iteration algorithm under the maximum entropy RL framework. 

### Features of this implementation



### Learning curves

Sometimes I get the weird runtime error:

ValueError: The parameter loc has invalid values
