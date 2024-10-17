from gymnasium import spaces
from pettingzoo import ParallelEnv

import numpy as np

class StochasticGame(ParallelEnv):  # also use from pettingzoo
    metadata = {'render_modes': ['human']}

    def __init__(self, reward_matrix, standard_deviations):
        self.agents = ["player_0", "player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.reward_matrix = reward_matrix
        self.standard_deviations = standard_deviations

        self.action_spaces = {agent: spaces.Discrete(reward_matrix.shape[0]) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Discrete(1) for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = None  # Stateless environment

        return {agent: 0 for agent in self.agents}  # Observations are dummy since the environment is stateless

    def step(self, actions):
        if not self.agents:
            return

        # Extract actions
        action_0 = actions[self.agents[0]]
        action_1 = actions[self.agents[1]]
        action_2 = actions[self.agents[2]]

        print("actions", action_0, action_1, action_2)

        # Get rewards from the reward matrix
        #print("actions", action_0, action_1, action_2)
        rewards_0, rewards_1, rewards_2 = self.get_rewards_for_actions(action_0, action_1, action_2)
        #print("rewards", rewards_0, rewards_1, rewards_2)

        # Update rewards
        self.rewards = {self.agents[0]: rewards_0, self.agents[1]: rewards_1, self.agents[2]: rewards_2}

        # Since this is a one-shot game, mark agents as done
        self.dones = {agent: True for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}

        # No additional info in this case
        self.infos = {agent: {} for agent in self.agents}

        # Observations are dummy since the environment is stateless
        observations = {agent: 0 for agent in self.agents}

        return observations, self.rewards, self.dones, self.truncateds, self.infos

    def get_rewards_for_actions(self, action_0, action_1, action_2):
        mean = self.reward_matrix[action_0, action_1, action_2]
        sigma = self.standard_deviations.get_sigma(action_1, action_2)

        # normal distribution with mean and sigma
        return mean + np.random.normal(0, sigma, 3)

    def render(self):
        pass

    def close(self):
        pass


class StandardDeviation:
    def __init__(self, sigma0, sigma1, sigma):
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.sigma = sigma

    def get_sigma(self, action_b, action_c):
        """
        Get the sigma that is supposed to be application as per our game rules
        """
        if action_b == 0 and action_c == 0:  # Corresponds to joint action <ai, b1, c1>
            return self.sigma0
        elif action_b == 1 and action_c == 1:  # Corresponds to joint action <ai, b2, c2>
            return self.sigma1
        else:  # For all other combinations
            return self.sigma
