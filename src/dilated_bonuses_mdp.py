'''
https://arxiv.org/abs/2107.08346
'''

import math
import numpy as np
import copy

from typing import Dict



class DBMDP():

    def __init__(self, policy, env, delta, H, T):
        '''
        Parameters:
            policy -- 
            env --
            delta --
            H -- the number of layers in the episodic MDP.
            T -- the number of epochs to train for
        '''
        # initialize environment 
        self._T = T

        self._env = env
        self._size_X = env.observation_space.n
        self._size_A = env.action_space.n

        self._delta = delta
        self._H = H
        self._eta = min(1/24 * H**3, 1/math.sqrt(self._size_X * self._size_A * H * T))
        self._gamma = 2 * self._eta * H
        
        # Initialize algorithm
        self._k = 1

        # initialize counters
        self._N_k_minus_1 = np.zeros((self._size_X, self._H, self._size_A))
        self._N_k = np.zeros((self._size_X, self._H, self._size_A))
        self._N_k_minus_1_transition = np.zeros((self._size_X, self._H, self._size_A, self._size_X, self._H))
        self._N_k_transition = np.zeros((self._size_X, self._H, self._size_A, self._size_X, self._H))

        # Initialize transition functions
        self._P_hat = None # TODO set, but not sure how yet
        # really is N_k / max(1, N_k), but since N_k is 0, we can save time and initialize to 0
        self._P_bar = np.zeros((self._size_X, self._H, self._size_A, self._size_X, self._H)) 
        # TODO is this right for our conficence set?
        self._epsilon = np.zeros((self._size_X, self._H, self._size_A, self._size_X, self._H)) 

        # stuff not explicityly mentioned to initialize
        # should initialize to 0 or infinity?
        self._Q_hat = np.zeros((self._size_X, self._H, self._size_A))
        self._B = np.zeros((self._size_X, self._H, self._size_A))
        self._Q_and_B_sum = self._Q_hat + self._B

        # TODO maybe need to make state for every state and every layer.
        # actually, I think this is the case, since we need to weight being in a state differently based on layer


        self._checkrep()

    def _checkrep(self):
        pass

    def _initialize_algorithm(self):
        '''
        Initializes internal state necessary to start algorithm training
        '''
        pass

    def _compute_policy(self):
        '''
        Computes a policy for all state and action pairs.

        Assume training step t s.t. 1 <= t <= T.  The polciy pi_t(a|x) is proportional to

        exp(-\eta sum_{tau = 1}^{t-1} (Q^{\hat}_{\tau} (x,a) - B_{\tau} (x,a)))

        Assumes:
        Q_hat corresponds to Q_{t-1}
        B corresponds to B_{t-1}

        Modifies:
        self._Q_and_B_sum to include self._Q_hat and self._B

        Returns:
        A |X| x H x |A| numpy array normalized such that \sum_{i=1}^{|A|} pi(a|x) = 1 for all x, h in X x H.
        '''
        # add in latest Q and B
        self._Q_and_B_sum += self._Q_hat + self._B

        # Multiply by eta
        scaled_sum = - self._eta * self._Q_and_B_sum

        # exponential
        proportion = np.exp(scaled_sum)

        # policy
        # sum along axis = 1 (actions axis) so we have a normalized policy for each state
        normalization = np.sum(proportion, axis=2)
        assert normalization.shape == (self._size_X, self._H)

        normalization = np.expand_dims(normalization, axis = 2)
        normalization = np.tile(normalization, (1,1,self._size_A))
        assert normalization.shape == (self._size_X, self._H, self._size_A)

        # TODO is this safe if actions and states are same size?
        return np.divide(proportion, normalization)

    def _select_action(self, policy, state_index):
        '''
        Selects an action from a given state and policy.

        Parameters:
            policy -- an |X| x H x |A| numpy array normalized such that 
                \sum_{i=1}^{|A|} pi(a|x) = 1 for all x.
            state_index -- an index of the policy for the given state of the form (x, h)
                where x is a member of X and 0 <= h < H.

        Returns:
        The index of the action to take.
        '''
        x, h = state_index
        assert 0 <= x < policy.shape[0]
        assert 0 <= h < policy.shape[1]

        # select action
        state_policy = policy[x, h, :]
        action_index = np.random.choice(len(state_policy), p=state_policy)

        return int(action_index)

    def _execute_policy(self, policy : np.ndarray):
        '''
        Executes a policy for all H episodes of the episodic MDP.

        Parameters:
            policy -- an |X| x h x |A| numpy array normalized such that 
                \sum_{i=1}^{|A|} pi(a|x) = 1 for all x, h.

        Returns: the trajecotry of the form (X, A, L) where 
        X is a length H array of observed states of the form (x,h), A is a length H array of actions, 
        and L is a length H array of observed rewards.
        '''
        observations = []
        actions = []
        rewards = []
        
        # for each level

        # TODO figure out indices (should the for loop start at 1)
        # TODO states should always come with level h

        # TODO need to always start at same state
        (obs, 0) = self._env.reset()
        for h in range(self._H):
            observations.append((obs, h))

            # TODO need to convert state and action numbers to actual
            # TODO it does work with discrete for both
            action = self._select_action(policy, obs)
            actions.append(action)

            obs, reward, done, info = self._env.step(action)
            rewards.append(reward)

        assert done
        assert len(observations) == len(rewards) == len(actions) == self._H
        return (observations, actions, rewards)

    def _compute_L_masked(self, observations, actions, rewards):
        '''
        Computes an array which, for each state and action pair visited, contains the rewards for that state and action
        plus all subsequent states and actions.

        More formally, define L_{t,h} = \sum_{i=h}^{H-1} rewards(i).  Then this function returns
        L_{t,h} 1_t{x_{t,h} = x, a_{t,h} = a} where 1 is the indicator vector function.

        Parameters:
            observations, actions, rewards obtained over a trajectory.

        Returns:
        A |X| x H x |A| ndarray containing L masked by the indicator function.
        '''
        assert len(observations) == len(rewards) == len(actions)
        L_masked = np.zeros((self._size_X, self._H, self._size_A))

        rewards_sum = 0
        # for L in reverse order
        for h in range(len(rewards), 0, -1):
            # rewards sum now equals rewards from level h onwards
            rewards_sum += rewards[h]
            
            action_at_h = actions[h]
            state_at_h = observations[h]
            level_at_h = h # TODO double check that this is true

            L_masked[state_at_h, level_at_h, action_at_h] = rewards_sum

        return L_masked

    def _find_sigma(self, f, k, n, reverse : bool) -> np.array:
        '''
        Finds a bijection sigma : [n] -> X in layer k such that 
            f(sigma(1)) <= f(sigma(2)) <= ... <= f(sigma(n))

        Parameters:
            f -- X x h -> [0,1] ; the upper bound on the probability of visiting some fixed state
                x from x_tilde in X
            k -- the layer index 0 <= k < self._H
            n -- the number of states in layer k
            reverse -- whether to reverse the sort order.  If true, the bijection becomes
                f(sigma(1)) >= f(sigma(2)) >= ... >= f(sigma(n))

        Returns:
        An array satisfying array[i] = q(i)
        '''
        f_k = f[:, k]

        sort = np.argsort(f_k)

        if reverse:
            return sort.flip()
        else:
            return sort


    def _greedy(self, f, P_bar, k, epsilon, reverse):
        '''
        Parameters:
            f -- X x h -> [0,1] ; the upper bound on the probability of visiting some fixed state
                x from x_tilde in X
            P_bar -- X -> [0, 1] a distribution over n states of layer k
            k -- the layer to solve for
            epsilon -- X -> positive numbers ; positive numbers {e(x)}_{x \in X_k}
            reverse -- If true, reverses the greedy from a maximization to a minimization

        '''
        # TODO are we supposed to modify in place here?
        P_bar = copy.deepcopy(P_bar)
        epsilon = copy.deepcopy(epsilon)

        # TODO assert shapes
        assert P_bar.shape == (self._size_X)

        # since we have n states in layer k
        n = self._size_X

        # initialize:
        # j^- = 1, j^+ = n, sort {f(x)}_{x \in X_k} and find \sigma such that
        # f(sigma(1)) <= f(sigma(2)) <= ... <= f(sigma(n))
        # change to 0, n-1 since 0 indexing
        j_minus = 0
        j_plus = n-1 # since num states the same for all layers

        sigma = self._find_sigma(f, k, n)

        # while j^- < j^+ do
        while j_minus < j_plus:
            # x_minus = sigma(j_minus), x^+ = sigma(j^+)
            x_minus = sigma[j_minus]
            x_plus = sigma[j_plus]
            # delta^minus = min(p_bar(x_minus), epsilon(x_minus))
            delta_minus = min(P_bar[x_minus], epsilon[x_minus])
            # delta^plus = min(1-p_bar(x^+, epsilon(x^+)))
            delta_plus = min(1-P_bar[x_plus], epsilon[x_plus])
            
            # p_bar(x^-) <- p_bar(x^-) - min(delta^- delta^+)
            P_bar[x_minus] = P_bar[x_minus] - min(delta_minus, delta_plus)
            # p_bar[x^+] <- p_bar(x^+) + min(delta^-, delta^+)
            P_bar[x_plus] = P_bar[x_plus] + min(delta_minus, delta_plus)

            # if delta_minus  <= delta_plus then
            if delta_minus <= delta_plus:
                # epsilon(x^+) <- epsilon(x^+) - delta^-
                epsilon[x_plus] = epsilon[x_plus] - delta_minus
                # j^- <- j^- +1
                j_minus += 1
            else:
                # epsilon(x^-) <- epsilon(x^-) - delta^+
                epsilon[x_minus] = epsilon[x_minus] - delta_plus
                j_plus -= 1

        return sum(P_bar[sigma[j]] * f[sigma[j], k] for j in range(n))




    def _comp_uob(self, policy, state, action, reverse):
        '''

        Based on Jin et al., 2020

        Parameters:
            policy -- an |X| x H x |A| numpy array normalized such that 
                \sum_{i=1}^{|A|} pi(a|x) = 1 for all x, h.
            state -- a state, level pair describing the state
            action -- an index into the policy giving an action
            P_bar -- an X x H x A x X x H confidence set
            epsilon -- an X x H x A x X x H set of confidence parameters
            reverse -- if True, reverses the algorithm to comp_lob
        '''
        x, h = state

        # initialize: for all x_tiddle in X_k(x), set f(x_tiddle) = 1(x_tiddle = x)
        # translation: for all x in X_h, set f(x) = 1{x_tiddle = x}
        # simpler : set f(x) = 1
        f_x = np.zeros((self._size_X, self._H))
        f_x[x, h] = 1

        # for k = k(x) - 1 to 0 do
        for k in range(h-1, 0, -1): # TODO h or h-1?
            # for all x_tilde in X_k do
            for x_tilde in range(self._size_A):
                # Compute f(x_tilde) based on
                # f(x_tilde) = \sum_{a \in A} \pi_t (a| x_tilde) * GREEDY(f, P_bar(*|x_tilde, a), epsilon(*|x_tilde, a), max)
                f_x[x_tilde, k] = np.sum(
                    policy[x_tilde, k, a] * self._greedy(f_x, self._P_bar, self._epsilon, reverse) for a in range(self._size_A)
                )
        
        # TODO not sure what x_0 means
        return 


    def _compute_q_bar(self, policy, observations, actions, reverse):
        '''
        Computes an upper occupancy bound.

        q_bar represents the maximum probabilty of visiting a particular state and action
        under policy.  It is computed by using the transition function in the set of all 
        transition functions that maximizes the probability that state s is visited and 
        action a is taken in state s.
        '''
        q_bar = np.zeros((self._size_X, self._H, self._size_A))

        # only need to update for state-action pairs we observed
        for i in range(len(observations)):
            obs = observations[i]
            action = actions[i]
            h = i
            
            q_bar[obs, h, action] = self._comp_uob(policy, obs, action, reverse)

        return q_bar
        
    def _compute_b(self, policy, q_bar, q_lower_bar):
        '''
        Parameters:
            policy -- an |X| x H x |A| numpy array normalized such that 
                \sum_{i=1}^{|A|} pi(a|x) = 1 for all x, h.
            q_bar -- 
        '''
        assert policy.shape == (self._size_X, self._H, self._size_A)

        # result should be X x H
        # TODO do we need to divide by n_actions? don't think so, since weighting
        q_bar_average = np.sum(np.multiply(policy, q_bar), axis=2)
        q_lower_bar_average = np.sum(np.multiply(policy, q_lower_bar), axis=2)

        b_t = (3*self._gamma * self._H + self._H * (q_bar_average) - q_lower_bar_average)/(q_bar_average + self._gamma)

        assert b_t.shape == (self._X, self._H)

        return b_t

    def _compute_B(self, policy, b):

        B_t = np.zeros((self._size_X, self._H, self._size_A))

        # TODO can we make this faster than a for loop?
        for state in range(self._size_X):
            for action in range(self._size_A):
                B_t[state, action] = b[state] + (1 + 1/self._H) * self._comp_uob(policy, state, action, reverse=False)

        # TODO assert B_t[x_H, a] = 0 for all a
        return B_t

    def _update_counters(self, observations, actions):
        '''
        Updates the internal counters based on the observations and actions taken.

        Parameters:
            observations -- a list of observed states
            actions -- a list of actions taken.  Requires len(actions) == len(observations)
        
        Modifies:
        N_k(x_{t,h}, a_{t,h}) += 1, N_k(x_{t,h}, a_{t,h}, x_{t, h+1}) += 1 for all x_t, a_t in observations, actions
        '''
        for h in range(self._H-1):
            obs = observations[h]
            action = actions[h]
            next_obs = observations[h+1]
            self._N_k[obs, h, action] += 1
            self._N_k_transition[obs, h, action, next_obs, h+1]

        h += 1
        # missed last one
        self._N_k[observations[h], h, actions[h]]

    def _is_time_to_increment_epoch(self, observations, actions):
        '''
        Determines if it is time to increment the epoch, according to the doubling
        epoch schedule.
        '''
        # if \exists h, N_k(x_{t,h}, a_{t,h}) >= max(1, 2N_{k-1}(x_{t,h}, a-{t,h}))
        for h in range(self._H):
            obs = observations[h]
            action = actions[h]

            if self._N_k[obs, h, action] >= max(1, 2*self._N_k_minus_1[obs, h, action]):
                return True
        return False
        
    def _update_P_bar(self):
        '''
        Updates the internal value for P_bar
        '''
        self._P_bar = np.divide(self._N_k_transition, np.maximum(1, self._N_k))

    def _update_epsilon(self):
        '''Updates the internal value for epsilon'''
        log_term = math.log(self._T * self._size_X * self._size_A / self._delta)
        first_term = 4 * np.sqrt(np.divide(self._P_bar*log_term, np.maximum(1, self.N_k)))

        second_term = 28 * log_term / (3 * np.maximum(1, self._N_k))

        self._epsilon = first_term + second_term


    def learn(self):
        '''
        Main learning algorithm.
        '''
        # for t = 1, 2, T do
        for t in range(self._T):

            # Step 1: Compute and execute policy
            # Execute pi_t for one episode
            pi_t = self._compute_policy()
            observations, actions, rewards = self._execute_policy(pi_t)

            # TODO now assume same action for all states, adjust later - seems like ok assumption with env
            
            # Step 2: Construct Q-function estimators
            # for all h in {0, ..., H-1} and (x,a) in X_h X A
            L_I = self._compute_L_masked(observations, actions, rewards)
            q_bar_t = self._compute_q_bar(pi_t, observations, actions, reverse=False)

            Q_hat_t = L_I / (q_bar_t + self._gamma)

            # Step 3: Construct bonus functions.
            # For all (x,a) \in X x A
            q_lower_bar_t = self._compute_q_bar(pi_t, observations, actions, reverse=True)
            b_t = self._compute_b(pi_t, q_bar_t, q_lower_bar_t)
            B_t = self._compute_B(pi_t, b_t)

            # Step 4: Update model estimation:
            # for all h < H, N_k(x_{t,h}, a_{t,h}) += 1, N_k(x_{t,h}, a_{t,h}, x_{t, h+1}) += 1
            self._update_counters(observations, actions)
            if self._is_time_to_increment_epoch(observations, actions):
                # increment epoch index k += 1 and copy counters: N_k <- N_{k-1}, N_k <- N_{k-1}
                self._k += 1
                self._N_k = copy.deepcopy(self._N_k_minus_1)
                self._N_k_transition = copy.deepcopy(self._N_k_minus_1_transition)

                # Compute empirical transition P^{bar}_k (x'|x,a) = N_k(x,a,x')/max(1, N_k(x,a)) and confidence set
                self._update_P_bar()

                # Update epsilon
                self._update_epsilon()






