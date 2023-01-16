from bron_gym.bron_gym import BRONEnv, BRONEnvEVOAPT, BRONEnvTactic
import numpy as np

'''
Learning Adversarial MDP's with Bandit Feedback and Unknown Transition
https://arxiv.org/abs/1912.01192
'''

def update_confidence_set(L, M, N, T, size_X, size_A, delta):
    '''
    delta -- confidence parameter
    '''
    P_low = {}
    P_high = {}
    # format of P: {k : np.arra((x, a, x'))}
    for k in range(L):
        M_i = M[k]
        N_i = N[k]

        P_bar = M_i / np.max(1, N_i) # TODO line up dimensions right.
        # need to divide each x' depth of M_i by N_i

        c = np.ln(T * size_X * size_A / delta)
        e_i = 2 * np.sqrt(P_bar * c / (np.max(1, N_i - 1))) + 14 * c / (3 * np.max(1, N_i -1))

        P_low[k] = P_bar - e_i
        P_high[k] = P_bar + e_i

    return (P_low, P_high)

def comp_uob(pi, x, a, P):
    raise NotImplementedError

def get_policy(q_hat):
    '''
    Gets the policy for an occupancy measure.

    Parameters:
        q_hat -- A dictionary containing keys k for 0 <= k < len(q_hat) where each key 
            contains a 3D numpy array of size |X_k| X |A_k| X |X_k+1|
    
    Returns:
    A policy of the form of a dictionary containing keys k for 0<=k<=len(q_hat)
    where policy[k] is a numpy array of size |X_k|X|A_k| where policy[i][j] represents 
    the probability of choosing action j at state i.
    '''
    pi = {}

    for k, q_k in q_hat.items():
        size_x, size_a, size_x_prime = q_k.shape
        normalization = np.repeat(np.sum(np.sum(q_hat, axis=1), axis=2), size_x_prime, axis=2)

        dist = np.sum(q_hat, axis=2)

        assert dist.shape == normalization.shape

        pi_k = np.true_divide(dist, normalization)
        pi[k] = pi_k
    return pi

def sample_policy(pi, k, s):
    '''
    Gets a random action for policy pi at layer k.

    Parameters:
    pi -- a policy as returned by get_policy
    k -- the layer to sample
    s -- the state number to sample from

    Returns:
    A randomly sampled action according to pi
    '''
    dist = pi[k][s]
    n_states, n_actions = pi[k].shape

    return np.random.choice(n_actions, p=dist)


def uob_reps(env, L, n_episodes, learning_rate, exploration_parameter, confidence_parameter):
    '''
    Parameters:
        env -- the OpenAI gym environment.  Must support the following operations:
            env.get_layer_states(k): returns an iterable of states in layer k.
            env.get_layer_actions(k): returns an iterable of actions available in layer k.
            env.get_state(): returns the current state
        L -- the number of layers in the MDP.  Requires L >= 2
        n_episodes -- the number of episodes to play through the MDP to learn.  Requires n_episodes > 0
        learning_rate -- 
        exploration_parameter --
        confidence_parameter -- 

    '''
    # initialize epoch index 1
    epoch_index = 1

    # initialize P_1 as set of all transition functions
    P_i = None # TODO
    # TODO what form is confidence parameter in?

    # for all k=0 to L-1 and all (x, a, x') in X_k x A x X_k+1
    # for k in range(L):
    # N is a dictionary containing entries k=0 to L-1
    # each entry is a |states at layer k| X |actions at layer k| array
    # M is same, except we arrays are now |states at layer k| X |actions at layer k| X |states at layer k+1|
    N_i_minus_1 = { k : np.zeros((len(env.get_all_states(k)), len(env.get_all_actions(k)))) for k in range(L) }
    N_i = { k : np.zeros((len(env.get_all_states(k)), len(env.get_all_actions(k))), L) for k in range(L) }
    M_i_minus_1 = { k : np.zeros((len(env.get_all_states(k)), len(env.get_all_actions(k)), len(env.get_all_states(k+1)), L)) for k in range(L-1) }
    M_i = { k : np.zeros((len(env.get_all_states(k)), len(env.get_all_actions(k)), len(env.get_all_states(k+1)))) for k in range(L-1) }

    # and occupancy measure
    # array of size |states at k=0| X |actions at k=0| X |states at k=1|
    q_hat_1 = {k: np.ones((len(env.get_all_states(k)), len(env.get_all_actions(k)), len(env.get_all_states(k+1)))) / (len(env.get_all_states(k)) * len(env.get_all_actions(k)) * len(env.get_all_states(k+1))) for k in range(L-1)}

    # initialize policy pi_1 = pi^{q_hat_1}
    # dictionary containing 0<=k<=L-1
    # , a numpy array of size |X_k| X |A_k| for each entry k
    pi = get_policy(q_hat_1)

    # for t=1 to T do
    for t in range(n_episodes):
        # reset environment
        env.reset()
        # execute policy for L steps and obtain trajectory x_k, a_k, l(x_k, a_k)
        x_realized = []
        a_realized = []
        l_realized = []
        loss_estimator = []
        for k in range(L):
            x_k = env.get_current_state()
            a_k = sample_policy(pi, k, x_k)

            observations, reward, done, info = env.step(a_k)
            x_k_plus_1 = None
            x_realized.append(x_k_plus_1) # TODO should x_k already be here?, not all indices the same
            a_realized.append(a_k)
            l_realized.append(reward)

        u_t = []
        for k in range(L):

            # compute upper occupancy bound for each k
            u_t.append(comp_uob(pi, x_k, a_k, P_i))

            # Construct loss estimator for all x, a
            # it's 0 for all actions not taken
            assert l_realized == reward
            loss_estimator.append(l_realized[k] / (u_t[k] + exploration_parameter))

            # Update counters, for each k
            N_i[k][x_k, a_k] += 1
            M_i[k][x_k, a_k, x_k_plus_1] += 1

        # If there exists k st. N_i (x_k, a_k) >= max(1, 2N_i-1(x_k, a_k))
        # TODO this doesnt seem right.  Seems like we increment epoch if only one x_k, a_k is incremented
        for k in range(L):
            if np.any(N_i[k] - np.max(1, 2* N_i_minus_1[k]) > 0):
                epoch_index += 1

                # Initialize new counters, for all x, a, x'
                # TODO seems backwards in paper, double check.  Flipped order here
                # TODO update counters for all k?
                N_i_minus_1 = N_i
                M_i_minus_1 = M_i

                # Update confidence set P_i based on equation 5
                P_i = update_confidence_set(P_i)

                # TODO right to break out of loop?
                break

        # Update occupancy measure D defined in equation 8

        # Update policy pi_t+1 = pi^q^t+1

    return pi
        

def learn(env, n_episodes):
    '''
    Learns an estimate of the optimal policy pi* over n_iterations.
    
    Uses the algorithm developed by Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, Tiancheng Yu
    found at https://arxiv.org/abs/1912.01192
    
    Returns:
    A representation of the optimal policy.
    '''
    # State space and action space defined by environment

    # L is number of CVEs that we need to pick
    L = 3
    # set initial policy: random
    policy = [env.action_space.sample() for i in range(L)]

    # for i=1 to T do:
    for i in range(n_episodes):

        env.reset()
        reward_i = 0

        # adversary decides loss function: set by env

        # learner decides a policy pi_t and starts in state x_o

        # for k = 0 to L-1 do:

        for k in range(L):

            # learner selects action
            observation, reward, done, info = env.step(policy[i])

            # learner observes loss
            reward_i += reward

            # environment draws new state - done above

            # learner observes state - done above


