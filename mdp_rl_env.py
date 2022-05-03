import gym

class MDPRLEnv(gym.env):

    def get_all_states(self, k):
        raise NotImplementedError

    def get_all_actions(self, k):
        raise NotImplementedError