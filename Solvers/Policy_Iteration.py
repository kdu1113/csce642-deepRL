# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
from Solvers.Abstract_Solver import AbstractSolver, Statistics

def get_random_policy(num_states, num_actions):
    policy = np.zeros([num_states, num_actions])
    for s_idx in range(num_states):
        action = s_idx % num_actions
        policy[s_idx, action] = 1
    return policy

class PolicyIteration(AbstractSolver):

    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)
        # Start with a random policy
        # self.policy[s,a] denotes \pi(a|s)
        # Note: Policy is determistic i.e., only one element in self.policy[s,:] is 1 rest are 0
        self.policy = get_random_policy(env.observation_space.n, env.action_space.n)

    def train_episode(self):
        """
            Run a single Policy iteration. Evaluate and improves the policy.

            Use:
                self.policy: [S, A] shaped matrix representing the policy.
                             self.policy[s,a] denotes \pi(a|s)
                             Note: Policy is determistic i.e., only one element in self.policy[s,:] is 1 rest are 0
                self.env: OpenAI environment.
                    env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                    env.nS is a number of states in the environment.
                    env.nA is a number of actions in the environment.
                self.options.gamma: Gamma discount factor.
                np.eye(self.env.action_space.n)[action]
        """

        # Evaluate the current policy
        self.policy_eval()

        # For each state...
        for s in range(self.env.observation_space.n):
            # Find the best action by one-step lookahead
            # Ties are resolved by returning the first action with maximum value (Hint: use max/argmax directly).

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            max_action = None
            max_value = float("-inf")

            for action in range(self.env.action_space.n):
                action_value = 0
                for prob, next_state, reward, done in self.env.P[s][action]:
                    action_value += prob * (reward + self.options.gamma * self.V[next_state])
                if action_value > max_value:
                    max_value = action_value
                    max_action = action

            # clear state s policy
            self.policy[s][:] = 0
            # set policy
            self.policy[s][max_action] = 1.0

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Policy Iteration"

    def one_step_lookahead(self, state):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A

    def policy_eval(self):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.
        Use a linear system solver sallied by numpy (np.linalg.solve)

        Use:
            self.policy: [S, A] shaped matrix representing the policy.
                         self.policy[s,a] denotes \pi(a|s)
                         Note: Policy is determistic i.e., only one element in self.policy[s,:] is 1 rest are 0
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            self.options.gamma: Gamma discount factor.
            np.linalg.solve(a, b) # Won't work with discount factor = 0!
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # Vπ(s) = ∑s` P(s`|s,π(s))(R(s,π(s),s`) + rV*(s`))
        # A = np.array(r*numof(s`) + ∑s` P(s`|s,π(s))(R(s,π(s),s`))
        # B = np.array([Vπ(s)])
        # V*(s`) = np.linalg.solve(A,B)
        # self.V[s`] = V*(s`)

        num_of_states = self.env.observation_space.n
        A = np.zeros((num_of_states, num_of_states))
        B = np.zeros(num_of_states)

        for state in range(num_of_states):
            # get action from current policy
            action = np.argmax(self.policy[state])
            # set coefficients for the equation
            for prob, next_state, reward, done in self.env.P[state][action]:
                A[state, next_state] += prob * self.options.gamma
                B[state] += prob * reward
        # solve equation
        self.V = np.linalg.solve(-np.identity(num_of_states) + A, -B)

    def create_greedy_policy(self):
        """
        Return the currently known policy.


        Returns:
            A function that takes an observation as input and greedy action as integer
        """
        def policy_fn(state):
            return np.argmax(self.policy[state])

        return policy_fn
