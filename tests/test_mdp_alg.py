import unittest
import mdp_alg
from bron_gym.bron_gym import BRONEnv, BRONEnvEVOAPT, BRONEnvTactic

'''Tests for the mdp_rl module.'''

# Testing strategy
# learn
#   env: 


class Testlearn(unittest.TestCase):
    '''Tests the learn function'''

    def test_learn_basic(self):
        network_description = {
            "nodes": {
                "c1": {
                    "apps": [
                        "cpe:2.3:a:symantec:norton_antivirus:*:*:*:*:*:*:*:*",
                    ],
                    "os": "cpe:2.3:o:microsoft:windows_7:-:*:*:*:*:*:*:*",
                },
            }
        }
        
        env = BRONEnvEVOAPT(rng_seed=1, network_description=network_description)
        print('original network', network_description)
        print('optimal policy', mdp_alg.learn(env, 2))

        


