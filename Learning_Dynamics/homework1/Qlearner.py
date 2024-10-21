import numpy as np
import random

class Qlearner:
    """A Q-learning agent"""

    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_size = action_size
        self.state_size = state_size

        # initialize the Q-table:
        self.qtable = np.zeros((self.state_size, self.action_size))

        # define learning rate:
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros((self.state_size, self.action_size))
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

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def select_greedy(self, state):
        # np.argmax(self.qtable[state]) will select first entry if two or more Q-values are equal, but we want true randomness:
        return np.random.choice(np.flatnonzero(np.isclose(self.qtable[state], self.qtable[state].max())))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = self.select_greedy(state)
        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def update(self, state, action, new_state, reward, done, update_epsilon=True):
        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            self.action_counter[state, action] += 1
            lr = 1 / self.action_counter[state, action]

        # Q(s,a) <-- Q(s,a) + learning_rate [R + gamma * max_a' Q(s',a') - Q(s,a)]
        # NOTE: in this case, you can safely remove self.gamma * np.max(self.qtable[new_state]), since the (not done) variable
        # before will always be 0, so it disappears. I left it in for uniformity of the code to other environments.
        self.qtable[state, action] += lr * (reward - self.qtable[state, action])

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
Independent learner class for 3
"""
class QLearner3x3x3OLD:
    """Q-learning agent for 3x3x3 stochastic game"""

    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_size = action_size
        self.state_size = state_size

        # Initialize the Q-table for independent actions
        self.qtable = np.zeros((self.state_size, self.action_size))

        # Define learning rate:
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros((self.state_size, self.action_size))
        else:
            self.dynamic_lr = False
            self.learning_rate = learning_rate

        # Discount factor:
        self.gamma = gamma

        # Exploration parameters:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tracking rewards/progress:
        self.rewards_this_episode = []  # During an episode, save every time step's reward
        self.episode_total_rewards = []  # Each episode, sum the rewards, possibly with a discount factor
        self.average_episode_total_rewards = []  # The average (discounted) episode reward to indicate progress

        self.state_history = []
        self.action_history = []

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def select_greedy(self, state):
        """Selects the action with the highest Q-value for the current state"""
        print("+++++++++++++++++ Qtable", self.qtable[state])
        print("Max", self.qtable[state].max())
        print("Flat nonzero", np.flatnonzero(np.isclose(self.qtable[state], self.qtable[state].max())))
        return np.random.choice(np.flatnonzero(np.isclose(self.qtable[state], self.qtable[state].max())))

    def select_action(self, state):
        """Selects an action using epsilon-greedy exploration strategy"""
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = self.select_greedy(state)
        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def update_epsilon(self):
        """Updates epsilon (exploration rate) after each episode"""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def update(self, state, action, new_state, reward, done, update_epsilon=True):
        """Performs the Q-learning update"""
        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            self.action_counter[state, action] += 1
            lr = 1 / self.action_counter[state, action]

        # Q(s,a) <-- Q(s,a) + learning_rate [R + gamma * max_a' Q(s',a') - Q(s,a)]
        print("Action", action)
        #self.qtable[state, action] += lr * (reward + (not done) * self.gamma * np.max(self.qtable[new_state]) - self.qtable[state, action])
        self.qtable[state, action] += lr * (
                    reward - self.qtable[state, action])
        print("Qtable after update", self.qtable)

        self.rewards_this_episode.append(reward)

        if done:
            # Track total reward:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            k = len(self.average_episode_total_rewards) + 1  # Amount of episodes that have passed
            self._calculate_average_episode_reward(k, episode_reward)

            if update_epsilon:
                self.update_epsilon()

            # Reset the rewards for the next episode:
            self.rewards_this_episode = []

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        """Calculates the total reward for the current episode"""
        if discount:
            return sum([self.gamma ** i * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        """Calculates the average reward over all episodes"""
        if k > 1:  # Running average is more efficient
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        """Prints rewards and optionally epsilon and Q-table"""
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        if print_epsilon:
            print("Epsilon:", self.epsilon)
        if print_q_table:
            print("Q-table: ", self.qtable)

class QLearner3x3x3:
    """Q-learning agent for 3x3x3 stochastic game with epsilon-greedy and Boltzmann exploration"""

    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.1,
        gamma=0.6,
        epsilon=1.0,
        epsilon_min=0.3,
        epsilon_decay=0.999,
        learning_rate_decay=1,
        temperature=1.0,
        temperature_min=0.2,
        temperature_decay=0.999,
    ):
        self.action_size = action_size
        self.state_size = state_size

        # Initialize the Q-table for independent actions
        self.qtable = np.zeros((self.state_size, self.action_size))

        # Learning rate and its decay factor
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        # Discount factor:
        self.gamma = gamma

        # Exploration parameters:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Exploration temperature for Boltzmann exploration:
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

        # Tracking rewards/progress:
        self.rewards_this_episode = []  # During an episode, save every time step's reward
        self.episode_total_rewards = []  # Each episode, sum the rewards, possibly with a discount factor
        self.average_episode_total_rewards = []  # The average (discounted) episode reward to indicate progress

        self.state_history = []
        self.action_history = []

    def reset_agent(self):
        """Resets the agent's Q-table, epsilon, learning rate, and temperature"""
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.epsilon = 1.0  # Reset epsilon if necessary
        self.learning_rate = 0.5  # Reset the learning rate if necessary
        self.temperature = 1.0  # Reset the temperature

    def select_action(self, state):
        """Selects an action using epsilon-greedy and Boltzmann exploration with numerical stability"""
        if np.random.rand() < self.epsilon:
            # Epsilon-greedy random action
            action = random.randrange(self.action_size)
        else:
            # Boltzmann exploration with numerical stability
            q_values = self.qtable[state]

            # Clip Q-values to prevent overflow
            q_values = np.clip(q_values, -500, 500)

            # Subtract max Q-value to avoid large exponents
            exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)

            # Compute probabilities with added epsilon for numerical stability
            probs = exp_q / (np.sum(exp_q) + 1e-10)

            # Sample an action based on the probabilities
            action = np.random.choice(self.action_size, p=probs)

        # Track the selected action
        self.state_history.append(state)
        self.action_history.append(action)

        return action

    def update_epsilon(self):
        """Updates epsilon (exploration rate) after each episode"""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        #print(f"Updated epsilon: {self.epsilon}")

    def update_temperature(self):
        """Updates temperature (exploration rate) after each episode"""
        self.temperature *= self.temperature_decay
        self.temperature = max(self.temperature_min, self.temperature)
        #print(f"Updated temperature: {self.temperature}")

    def update_learning_rate(self):
        """Updates the learning rate after each episode"""
        self.learning_rate *= self.learning_rate_decay
        #print(f"Updated learning rate: {self.learning_rate}")

    def update(self, state, action, new_state, reward, done, update_epsilon=True):
        """Performs the Q-learning update with clipped negative rewards"""

        # Clip rewards so they don't go below -10 (adjust as necessary)
        clipped_reward = np.clip(reward, -10, None)

        # Q(s,a) update rule
        self.qtable[state, action] = (1 - self.learning_rate) * self.qtable[state, action] + self.learning_rate * (
                clipped_reward + (not done) * self.gamma * np.max(self.qtable[new_state])
        )

        self.rewards_this_episode.append(reward)

        if done:
            # Track total reward:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            if update_epsilon:
                self.update_epsilon()

            # Update learning rate and temperature after the episode
            self.update_learning_rate()
            self.update_temperature()

            # Reset the rewards for the next episode:
            self.rewards_this_episode = []

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        """Calculates the total reward for the current episode"""
        if discount:
            return sum([self.gamma ** i * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        """Calculates the average reward over all episodes"""
        if k > 1:  # Running average is more efficient
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)

    def print_rewards(self, episode, print_epsilon=True, print_temperature=True, print_q_table=True):
        """Prints rewards and optionally epsilon, temperature, and Q-table"""
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        if print_epsilon:
            print("Epsilon:", self.epsilon)
        if print_temperature:
            print("Temperature:", self.temperature)
        if print_q_table:
            print("Q-table: ", self.qtable)
