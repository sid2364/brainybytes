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
 the simple Boltzmann action selection.  Make sure the temperature decays 
slowly to eventually reach a fully exploiting policy.

This class is written to handle 3 agents, self and two opponents. 
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
            temperature_min=0.01,
            temperature_decay=0.995,
    ):
        self.action_size = action_size
        self.opponent1_action_size = opponent_action_size
        self.opponent2_action_size = opponent_action_size # same as opponent 1
        self.state_size = state_size

        # Initialize Q-table for joint actions (state, own action, opponent 1, opponent 2)
        self.qtable = np.zeros(
            (self.state_size, self.action_size, self.opponent1_action_size, self.opponent2_action_size))

        # Track opponents' joint actions
        self.opponent1_action_counts = np.zeros(self.opponent1_action_size)
        self.opponent2_action_counts = np.zeros(self.opponent2_action_size)

        self.opponent_joint_action_distribution = np.ones((self.opponent1_action_size, self.opponent2_action_size)) / (
                self.opponent1_action_size * self.opponent2_action_size)
        self.opponent_joint_action_counts = np.zeros((self.opponent1_action_size, self.opponent2_action_size))
        self.total_observations = 0

        # Learning rate setup
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros(
                (self.state_size, self.action_size, self.opponent1_action_size, self.opponent2_action_size))
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

    def select_action2(self, state):
        # Calculate values as usual
        values = np.sum(self.qtable[state] * self.opponent_joint_action_distribution, axis=(1, 2))
        # print("qtable[state]", self.qtable[state])
        # print("opponent_joint_action_distribution", self.opponent_joint_action_distribution)
        # print("sum", np.sum(self.qtable[state] * self.opponent_joint_action_distribution, axis=(1, 2)))
        print("values", values)

        # Stabilize the softmax using log-sum-exp trick
        max_value = np.max(values)
        exp_values = np.exp((values - max_value) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        # print("probabilities", probabilities)

        # Choose an action based on the probabilities
        action = np.random.choice(self.action_size, p=probabilities)
        # print("action", action)
        return action

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            expected_q_values = np.sum(
                self.qtable[state] * self.opponent_joint_action_distribution, axis=(1, 2)
            )

            # Step 2: Apply Boltzmann (softmax) to the marginalized Q-values for your agent's actions
            max_q_value = np.max(expected_q_values)  # For numerical stability
            exp_q_values = np.exp((expected_q_values - max_q_value) / self.temperature)
            #print("exp_q_values", exp_q_values)

            # Step 3: Compute action probabilities using softmax
            action_probabilities = exp_q_values / np.sum(exp_q_values)
            print("action_probabilities", action_probabilities)

            # Step 4: Sample an action based on the Boltzmann distribution
            action = np.random.choice(self.action_size, p=action_probabilities)

        return action

    def update_opponent_distribution2(self, opponent1_action, opponent2_action):
        # Track joint opponent actions
        self.opponent1_action_history.append(opponent1_action)
        self.opponent2_action_history.append(opponent2_action)

        self.opponent1_action_counts[opponent1_action] += 1
        self.opponent2_action_counts[opponent2_action] += 1

        # Update the joint action distribution
        print("opponent1_action_counts", self.opponent1_action_counts)
        print("opponent2_action_counts", self.opponent2_action_counts)
        joint_action_counts = np.outer(self.opponent1_action_counts, self.opponent2_action_counts)
        print("joint_action_counts", joint_action_counts)
        self.opponent_joint_action_distribution = joint_action_counts / np.sum(joint_action_counts)
        print("opponent_joint_action_distribution", self.opponent_joint_action_distribution)

    def update_opponent_distribution(self, opponent1_action, opponent2_action):
        # Increment the count for the observed joint action
        self.opponent_joint_action_counts[opponent1_action, opponent2_action] += 1
        self.total_observations += 1

        # Normalize counts to compute the new distribution
        self.opponent_joint_action_distribution = self.opponent_joint_action_counts / self.total_observations
        print("opponent_joint_action_distribution", self.opponent_joint_action_distribution)

    def update(self, state, action, opponent1_action, opponent2_action, new_state, reward, done, update_epsilon=True):
        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            # Dynamic learning rate for joint actions
            self.action_counter[state, action, opponent1_action, opponent2_action] += 1
            lr = 1 / self.action_counter[state, action, opponent1_action, opponent2_action]

            print("lr", lr)



        # print everything to debug

        """
        print("state, action, opponent1_action, opponent2_action", state, action, opponent1_action, opponent2_action)
        print("reward", reward)
        print("self.qtable[state, action, opponent1_action, opponent2_action]", self.qtable[state, action, opponent1_action, opponent2_action])
        print("lr", lr)
        print("reward - self.qtable[state, action, opponent1_action, opponent2_action]", reward - self.qtable[state, action, opponent1_action, opponent2_action])
        print("lr * (reward - self.qtable[state, action, opponent1_action, opponent2_action])",
              lr * (
                      reward - self.qtable[state, action, opponent1_action, opponent2_action])
              )
        print("self.qtable[state]", self.qtable[state])
        print("*" * 50)"""
        # Update Q-value for joint actions
        self.qtable[state, action, opponent1_action, opponent2_action] += lr * (
                reward - self.qtable[state, action, opponent1_action, opponent2_action])
        #print everything after updating
        #print("self.qtable[state] AFTER", self.qtable)
        #print("*" * 50)


        #self.qtable[state, action, opponent_action] += lr * (reward - self.qtable[state, action, opponent_action])
        #print("Q-table: ", self.qtable)

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

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        # print("Episode ", episode + 1)
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        print("Epsilon:", self.epsilon) if print_epsilon else None
        print("Q-table: ", self.qtable) if print_q_table else None
        print("Temperature:", self.temperature)