import numpy as np
import gym

class ScalingActionWrapper(gym.ActionWrapper):

    """Assumes that actions are symmetric about zero!!!"""

    def __init__(self, env, scaling_factors: np.array):
        super(ScalingActionWrapper, self).__init__(env)
        self.scaling_factors = scaling_factors

    def action(self, action):
        return self.scaling_factors * action
