import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv


class MatrixGame(ParallelEnv):
    metadata = {'render_modes': ['human']}

    def __init__(self, reward_matrix):
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.reward_matrix = reward_matrix

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

        # Get rewards from the reward matrix
        reward_0, reward_1 = self.reward_matrix[action_0, action_1]

        # Update rewards
        self.rewards = {self.agents[0]: reward_0, self.agents[1]: reward_1}

        # Since this is a one-shot game, mark agents as done
        self.dones = {agent: True for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}

        # No additional info in this case
        self.infos = {agent: {} for agent in self.agents}

        # Observations are dummy since the environment is stateless
        observations = {agent: 0 for agent in self.agents}

        return observations, self.rewards, self.dones, self.truncateds, self.infos

    def render(self):
        pass

    def close(self):
        pass
