import unittest

from gym.envs.toy_text import frozen_lake
from src import dilated_bonuses_mdp

class TestDBMDP(unittest.TestCase):

    def test_alg_simple(self):
        env = frozen_lake.FrozenLakeEnv()

        mdp = dilated_bonuses_mdp.DBMDP(None, env, 0.5, 10, 100)
        print('hi')
        mdp.learn()

