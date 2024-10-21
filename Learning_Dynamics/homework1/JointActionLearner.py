import numpy as np
import random


"""
Written for 2 agents, self and the opponent.
"""
class JointActionLearner:
    """A joint action learner"""

    def __init__(
        self,
        action_size,
        opponent_action_size,
        state_size,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_size = action_size
        self.opponent_action_size = opponent_action_size
        self.state_size = state_size

        # here we keep our belief over the opponent's strategy:
        self.opponent_action_counts = np.zeros(self.opponent_action_size)
        self.opponent_action_distribution = np.ones(self.opponent_action_size) / self.opponent_action_size  # initialize uniformly

        # initialize the Q-table:
        self.qtable = np.zeros((self.state_size, self.action_size, self.opponent_action_size))

        # define learning rate:
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros((self.state_size, self.action_size, self.opponent_action_size))
        else:
            self.dynamic_lr = False
            self.learning_rate = learning_rate

        # discount factor:
        self.gamma = gamma

        # Exploration parameters:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # tracking rewards/progress:
        self.rewards_this_episode = []  # during an episode, save every time step's reward
        self.episode_total_rewards = []  # each episode, sum the rewards, possibly with a discount factor
        self.average_episode_total_rewards = []  # the average (discounted) episode reward to indicate progress

        self.state_history = []
        self.action_history = []
        self.opponent_action_history = []

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size, self.opponent_action_size))
        self.opponent_action_counts = np.zeros(self.opponent_action_size)
        self.opponent_action_distribution = np.ones(self.opponent_action_size) / self.opponent_action_size

    def select_greedy(self, values):
        # np.argmax will select first entry if two or more Q-values are equal, but we want true randomness:
        max_value = np.max(values)
        max_indices = np.flatnonzero(np.isclose(values, max_value))
        return np.random.choice(max_indices)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # calculate values of each action weighted by opponent_action_distribution:
            values = np.sum(self.qtable[state] * self.opponent_action_distribution, axis=1)
            action = self.select_greedy(values)

        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def update_opponent_distribution(self, opponent_action):
        self.opponent_action_history.append(opponent_action)
        self.opponent_action_counts[opponent_action] += 1
        self.opponent_action_distribution = self.opponent_action_counts / np.sum(self.opponent_action_counts)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def update(self, state, action, opponent_action, new_state, reward, done, update_epsilon=True):
        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            self.action_counter[state, action, opponent_action] += 1
            lr = 1 / self.action_counter[state, action, opponent_action]

        # Q(s,a) <-- Q(s,a) + learning_rate [R + gamma * max_a' Q(s',a') - Q(s,a)]
        # Note: I removed (not done) * gamma * max_a' Q(s',a') since it has no use here to compute, as it is negated by the (not done) statement
        self.qtable[state, action, opponent_action] += lr * (reward - self.qtable[state, action, opponent_action])

        self.rewards_this_episode.append(reward)

        if done:
            # track total reward:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            k = len(self.average_episode_total_rewards) + 1  # amount of episodes that have passed
            self._calculate_average_episode_reward(k, episode_reward)

            if update_epsilon:
                self.update_epsilon()

            # reset the rewards for the next episode:
            self.rewards_this_episode = []

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        if discount:
            return sum([self.gamma**i * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        if k > 1:  # running average is more efficient:
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        # print("Episode ", episode + 1)
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        print("Epsilon:", self.epsilon) if print_epsilon else None
        print("Q-table: ", self.qtable) if print_q_table else None


"""
Extend the logic in JointActionLearner class to implement BoltzmannJointActionLearner that uses 
 the simple Boltzmann action selection.

This class is written to handle 3 agents, self and two opponents. 
"""
"""
Now defining the class which will play this game of 3x3x3 using the Boltzmann Action Function
"""


class BoltzmannJointActionLearner:
    def __init__(
            self,
            action_size,
            opponent_action_size,
            state_size,
            learning_rate=0.0,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            temperature=1.0,
            temperature_min=0.1,
            temperature_decay=0.995,
            agent_name="player",
            decay_factor=0.6
    ):
        self.name = agent_name
        self.action_size = action_size
        self.opponent_action_size = opponent_action_size
        self.state_size = state_size

        # Initialize Q-table for joint actions (state, own action, opponent1, opponent2)
        self.qtable = np.zeros(
            (self.state_size, self.action_size, self.opponent_action_size, self.opponent_action_size))

        # Track each opponent's actions separately
        self.opponent1_action_counts = np.zeros(self.opponent_action_size)
        self.opponent2_action_counts = np.zeros(self.opponent_action_size)

        self.opponent1_action_distribution = np.ones(self.opponent_action_size) / self.opponent_action_size
        self.opponent2_action_distribution = np.ones(self.opponent_action_size) / self.opponent_action_size

        self.decay_factor = decay_factor

        # Learning rate setup
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros(
                (self.state_size, self.action_size, self.opponent_action_size, self.opponent_action_size))
        else:
            self.dynamic_lr = False
            self.learning_rate = learning_rate

        # Parameters for exploration and Boltzmann action selection
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

        # Tracking rewards and progress
        self.rewards_this_episode = []
        self.episode_total_rewards = []
        self.average_episode_total_rewards = []

        self.state_history = []
        self.action_history = []
        self.opponent1_action_history = []
        self.opponent2_action_history = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
            #print(f"Randomly selected action: {action}")
        else:
            # Compute joint distribution as the outer product of individual distributions

            ## THIS IS WHERE THE PROBLEM OCCURS

            joint_distribution = np.outer(self.opponent1_action_distribution, self.opponent2_action_distribution)
            # print(f"Opponent1 Distribution: {self.opponent1_action_distribution*100}")
            # print(f"Opponent2 Distribution: {self.opponent2_action_distribution*100}")
            # print(f"Joint Distribution:\n{joint_distribution*100}")

            # Calculate expected Q-values for each of the agent's actions
            expected_q_values = np.sum(
                self.qtable[state] * joint_distribution,
                axis=(1, 2)
            )
            # print(f"Q-table at state {state}:\n{self.qtable[state]}")
            # print(f"Expected Q-values: {expected_q_values}")

            # Select the action with the highest expected Q-value
            """max_q = np.max(expected_q_values)
            max_actions = np.flatnonzero(expected_q_values == max_q)
            # print(f"Max Actions: {max_actions}")
            action = np.random.choice(max_actions)  # Break ties randomly
            """
            max_q_value = np.max(expected_q_values)  # For numerical stability
            exp_q_values = np.exp((expected_q_values - max_q_value) / self.temperature)
            action_probabilities = exp_q_values / np.sum(exp_q_values)
            # print(f"Action probabilities after softmax for {self.name}: {action_probabilities}")

            # Sample an action based on the Boltzmann distribution
            action = np.random.choice(self.action_size, p=action_probabilities)
            # print(f"Greedy selected action: {action}")

        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def select_action_new_old(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # Sample opponent actions directly from their distributions
            opponent1_action = np.random.choice(self.opponent_action_size, p=self.opponent1_action_distribution)
            opponent2_action = np.random.choice(self.opponent_action_size, p=self.opponent2_action_distribution)

            # Use the sampled opponent actions to compute expected Q-values
            expected_q_values = self.qtable[state, :, opponent1_action, opponent2_action]

            # Select the action with the highest expected Q-value
            max_q = np.max(expected_q_values)
            max_actions = np.flatnonzero(expected_q_values == max_q)
            action = np.random.choice(max_actions)  # Break ties randomly

        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def select_action_old(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
            # print(f"Randomly selected action: {action}")
        else:
            # Compute joint distribution as the outer product of individual distributions
            joint_distribution = np.outer(self.opponent1_action_distribution, self.opponent2_action_distribution)
            # print(f"ACTION SELECTION for {self.name}:")
            # print(f"Opponent1 Distribution: {self.opponent1_action_distribution}")
            # print(f"Opponent2 Distribution: {self.opponent2_action_distribution}")
            # print(f"Joint Distribution:\n{joint_distribution}")

            # Calculate expected Q-values for each of the agent's actions

            # print(f"Q-table at state {state}:\n{self.qtable[state]}")
            expected_q_values = np.sum(
                self.qtable[state] * joint_distribution,
                axis=(1, 2)
            )
            # print(f"Expected Q-values before softmax: {expected_q_values}")

            # Apply Boltzmann (softmax) to the expected Q-values
            max_q_value = np.max(expected_q_values)  # For numerical stability
            exp_q_values = np.exp((expected_q_values - max_q_value) / self.temperature)
            action_probabilities = exp_q_values / np.sum(exp_q_values)
            # print(f"Action probabilities after softmax for {self.name}: {action_probabilities}")

            # Sample an action based on the Boltzmann distribution
            action = np.random.choice(self.action_size, p=action_probabilities)
            # print(f"Selected action for {self.name}: {action}")

        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def update_opponent_distribution(self, opponent1_action, opponent2_action):
        # Update counts for each opponent independently
        self.opponent1_action_history.append(opponent1_action)
        self.opponent2_action_history.append(opponent2_action)

        self.opponent1_action_counts *= self.decay_factor
        self.opponent2_action_counts *= self.decay_factor

        self.opponent1_action_counts[opponent1_action] += 1
        self.opponent2_action_counts[opponent2_action] += 1

        # Update individual action distributions
        self.opponent1_action_distribution = self.opponent1_action_counts / np.sum(self.opponent1_action_counts)
        self.opponent2_action_distribution = self.opponent2_action_counts / np.sum(self.opponent2_action_counts)

    def update(self, state, action, opponent1_action, opponent2_action, new_state, reward, done, update_epsilon=True):
        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            # Dynamic learning rate for joint actions
            self.action_counter[state, action, opponent1_action, opponent2_action] += 1
            lr = 1 / self.action_counter[state, action, opponent1_action, opponent2_action]
            # print(f"Dynamic Learning Rate: {lr}")

        # Update Q-value for joint actions
        self.qtable[state, action, opponent1_action, opponent2_action] += lr * (
                reward - self.qtable[state, action, opponent1_action, opponent2_action])
        # print(
        #     f"Updated Q-table at state {state}, action {action}, opponent1 {opponent1_action}, opponent2 {opponent2_action}: {self.qtable[state, action, opponent1_action, opponent2_action]}")

        self.rewards_this_episode.append(reward)

        if done:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            k = len(self.average_episode_total_rewards) + 1
            self._calculate_average_episode_reward(k, episode_reward)

            if update_epsilon:
                self.update_epsilon()

            # Reset rewards
            self.rewards_this_episode = []

        self.update_temperature()

    def update_temperature(self):
        self.temperature *= self.temperature_decay
        self.temperature = max(self.temperature_min, self.temperature)
        # print(f"Updated Temperature: {self.temperature}")

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        if discount:
            return sum([self.gamma ** i * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        if k > 1:  # running average is more efficient:
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)
        # print(f"Average Episode Reward after {k} episodes: {average_episode_reward}")

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # print(f"Updated Epsilon: {self.epsilon}")

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        print("*****\nAGENT ", self.name)
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        if print_epsilon:
            print("Epsilon:", self.epsilon)
        if print_q_table:
            print("Q-table:", self.qtable)
        print("Temperature:", self.temperature)
