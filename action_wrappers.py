
import gym

class AlgoToEnvActionScalingWrapper(gym.ActionWrapper):

    def __init__(self, env, scaling_factor):
        super(AlgoToEnvActionScalingWrapper, self).__init__(env)
        self.scaling_factor = scaling_factor

    def action(self, action):
        return self.scaling_factor * action