# coding: utf-8

from MatrixGame import MatrixGame
from JointActionLearner import JointActionLearner
from Qlearner import Qlearner

import matplotlib.pyplot as plt
import numpy as np


"""
# Define the Hill-Climbing game reward matrix
ASYMMETRIC_COORDINATION_GAME = np.array([[(11, 11), (0, 0)],  # Both pick action 0 -> high reward
                                         [(0, 0), (6, 6)]])   # Both pick action 1 -> moderate reward

# Define the optimal joint action (0, 0) is optimal in the Hill-Climbing game
optimal_joint_action = (0, 0)

# Number of trial runs
num_trials = 100
num_episodes = 1000

# Store the probability of optimal joint action for each episode, averaged across trials
optimal_action_probs_jal = np.zeros(num_episodes)
optimal_action_probs_il = np.zeros(num_episodes)

# Run multiple trial runs
for trial in range(num_trials):
    # Initialize environment and agents for Joint Action Learners (JALs)
    env = MatrixGame(reward_matrix=ASYMMETRIC_COORDINATION_GAME)

    agent_0_jal = JointActionLearner(
        action_size=2,
        opponent_action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    agent_1_jal = JointActionLearner(
        action_size=2,
        opponent_action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # Keep track of the number of optimal actions for each episode
    optimal_actions_jal = np.zeros(num_episodes)

    for episode in range(num_episodes):
        observations = env.reset()
        state = 0  # Single state in the Hill-Climbing game

        done = False

        # Agents take actions
        action_0_jal = agent_0_jal.select_action(state)
        action_1_jal = agent_1_jal.select_action(state)

        actions = {"player_0": action_0_jal, "player_1": action_1_jal}

        # Step in the environment
        observations, rewards, dones, _, infos = env.step(actions)

        # Check if the agents selected the optimal joint action
        if (action_0_jal, action_1_jal) == optimal_joint_action:
            optimal_actions_jal[episode] += 1

        # Update the Q-tables
        agent_0_jal.update(state, action_0_jal, action_1_jal, state, rewards["player_0"], dones["player_0"])
        agent_1_jal.update(state, action_1_jal, action_0_jal, state, rewards["player_1"], dones["player_1"])

        # Update belief about the opponent's strategy
        agent_0_jal.update_opponent_distribution(action_1_jal)
        agent_1_jal.update_opponent_distribution(action_0_jal)

    # Calculate the probability of optimal joint action over episodes for this trial
    optimal_action_probs_jal += optimal_actions_jal / num_trials  # Averaging across trials


# In[7]:


# Plot the results
plt.plot(optimal_action_probs_jal, label='Joint Action Learners (JAL)')
plt.xlabel('Number of Interactions')
plt.ylabel('Probability of Choosing an Optimal Joint Action')
plt.title('Choosing Optimal Joint Actions')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


# Now, run the Independent Learner experiment
for trial in range(num_trials):
    # Initialize environment and agents for Independent Learners (ILs)
    env = MatrixGame(reward_matrix=ASYMMETRIC_COORDINATION_GAME)

    agent_0_il = Qlearner(
        action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    agent_1_il = Qlearner(
        action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # Keep track of the number of optimal actions for each episode
    optimal_actions_il = np.zeros(num_episodes)

    for episode in range(num_episodes):
        observations = env.reset()
        state = 0  # Single state in the Hill-Climbing game

        done = False

        # Agents take actions
        action_0_il = agent_0_il.select_action(state)
        action_1_il = agent_1_il.select_action(state)

        actions = {"player_0": action_0_il, "player_1": action_1_il}

        # Step in the environment
        observations, rewards, dones, _, infos = env.step(actions)

        # Check if the agents selected the optimal joint action
        if (action_0_il, action_1_il) == optimal_joint_action:
            optimal_actions_il[episode] += 1

        # Update Q-tables
        agent_0_il.update(state, action_0_il, state, rewards["player_0"], dones["player_0"])
        agent_1_il.update(state, action_1_il, state, rewards["player_1"], dones["player_1"])

    # Calculate the probability of optimal joint action over episodes for this trial
    optimal_action_probs_il += optimal_actions_il / num_trials  # Averaging across trials

# Plot the results with both JAL and IL
plt.plot(optimal_action_probs_jal, label='Joint Action Learners (JAL)')
plt.plot(optimal_action_probs_il, label='Independent Learners (IL)')
plt.xlabel('Number of Interactions')
plt.ylabel('Probability of Choosing an Optimal Joint Action')
plt.title('Choosing Optimal Joint Actions')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_values_heatmap(agent, agent_name):
    for state in range(agent.state_size):
        plt.figure(figsize=(8, 6))
        q_values = agent.qtable[state]  # Get the Q-values for this state
        sns.heatmap(q_values, annot=True, cmap="coolwarm", cbar=True, xticklabels=[f"Opponent Action {i}" for i in range(agent.opponent_action_size)], yticklabels=[f"Action {i}" for i in range(agent.action_size)])
        plt.title(f"Q-Values for {agent_name} in State {state}")
        plt.xlabel("Opponent Actions")
        plt.ylabel("Agent Actions")
        plt.show()

# Plot heatmaps for both JAL agents
plot_q_values_heatmap(agent_0_jal, "Agent 0 (JAL)")
plot_q_values_heatmap(agent_1_jal, "Agent 1 (JAL)")


# In[10]:


# Plot heatmaps for IL agents:
def plot_q_values_heatmap_il(agent, agent_name):
    for state in range(agent.state_size):
        plt.figure(figsize=(8, 6))
        q_values = agent.qtable[state]  # Get the Q-values for this state
        sns.heatmap(q_values.reshape(-1, 1), annot=True, cmap="coolwarm", cbar=True,
                    yticklabels=[f"Action {i}" for i in range(agent.action_size)],
                    xticklabels=[f"State {state}"])
        plt.title(f"Q-Values for {agent_name} in State {state}")
        plt.xlabel("State")
        plt.ylabel("Agent Actions")
        plt.show()

plot_q_values_heatmap_il(agent_0_il, "Agent 0 (IL)")
plot_q_values_heatmap_il(agent_1_il, "Agent 1 (IL)")


# In[11]:


# Both coordinated actions (0, 0) and (1, 1) give reward 10
COORDINATION_MATRIX = np.array([[(10, 10), (0, 0)],  # (0, 0) is optimal
                                  [(0, 0), (10, 10)]])  # (1, 1) is also optimal

# Number of trial runs
num_trials = 100
num_episodes = 1000

# Store the probability of optimal joint action for each episode, averaged across trials
optimal_action_probs_jal = np.zeros(num_episodes)
optimal_action_probs_il = np.zeros(num_episodes)

# Define the two optimal joint actions
optimal_joint_actions = [(0, 0), (1, 1)]

# Run multiple trial runs for JAL
for trial in range(num_trials):
    # Initialize environment and agents for Joint Action Learners (JALs)
    env = MatrixGame(reward_matrix=COORDINATION_MATRIX)

    agent_0_jal = JointActionLearner(
        action_size=2,
        opponent_action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    agent_1_jal = JointActionLearner(
        action_size=2,
        opponent_action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # Keep track of the number of optimal actions for each episode
    optimal_actions_jal = np.zeros(num_episodes)

    for episode in range(num_episodes):
        observations = env.reset()
        state = 0  # Single state in the Hill-Climbing game

        done = False

        # Agents take actions
        action_0_jal = agent_0_jal.select_action(state)
        action_1_jal = agent_1_jal.select_action(state)

        actions = {"player_0": action_0_jal, "player_1": action_1_jal}

        # Step in the environment
        observations, rewards, dones, _, infos = env.step(actions)

        # Check if the agents selected one of the optimal joint actions
        if (action_0_jal, action_1_jal) in optimal_joint_actions:
            optimal_actions_jal[episode] += 1

        # Update the Q-tables
        agent_0_jal.update(state, action_0_jal, action_1_jal, state, rewards["player_0"], dones["player_0"])
        agent_1_jal.update(state, action_1_jal, action_0_jal, state, rewards["player_1"], dones["player_1"])

        # Update belief about the opponent's strategy
        agent_0_jal.update_opponent_distribution(action_1_jal)
        agent_1_jal.update_opponent_distribution(action_0_jal)

    # Calculate the probability of optimal joint action over episodes for this trial
    optimal_action_probs_jal += optimal_actions_jal / num_trials  # Averaging across trials


# Now, run the Independent Learner (IL) experiment
for trial in range(num_trials):
    # Initialize environment and agents for Independent Learners (ILs)
    env = MatrixGame(reward_matrix=COORDINATION_MATRIX)

    agent_0_il = Qlearner(
        action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    agent_1_il = Qlearner(
        action_size=2,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # Keep track of the number of optimal actions for each episode
    optimal_actions_il = np.zeros(num_episodes)

    for episode in range(num_episodes):
        observations = env.reset()
        state = 0  # Single state in the Hill-Climbing game

        done = False

        # Agents take actions
        action_0_il = agent_0_il.select_action(state)
        action_1_il = agent_1_il.select_action(state)

        actions = {"player_0": action_0_il, "player_1": action_1_il}

        # Step in the environment
        observations, rewards, dones, _, infos = env.step(actions)

        # Check if the agents selected one of the optimal joint actions
        if (action_0_il, action_1_il) in optimal_joint_actions:
            optimal_actions_il[episode] += 1

        # Update Q-tables
        agent_0_il.update(state, action_0_il, state, rewards["player_0"], dones["player_0"])
        agent_1_il.update(state, action_1_il, state, rewards["player_1"], dones["player_1"])

    # Calculate the probability of optimal joint action over episodes for this trial
    optimal_action_probs_il += optimal_actions_il / num_trials  # Averaging across trials

# Plot the results with both JAL and IL
plt.plot(optimal_action_probs_jal, label='Joint Action Learners (JAL)')
plt.plot(optimal_action_probs_il, label='Independent Learners (IL)')
plt.xlabel('Number of Interactions')
plt.ylabel('Probability of Choosing an Optimal Joint Action')
plt.title('Choosing Optimal Joint Actions (Both Actions Optimal)')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_values_heatmap(agent, agent_name):
    for state in range(agent.state_size):
        plt.figure(figsize=(8, 6))
        q_values = agent.qtable[state]  # Get the Q-values for this state
        sns.heatmap(q_values, annot=True, cmap="coolwarm", cbar=True, xticklabels=[f"Opponent Action {i}" for i in range(agent.opponent_action_size)], yticklabels=[f"Action {i}" for i in range(agent.action_size)])
        plt.title(f"Q-Values for {agent_name} in State {state}")
        plt.xlabel("Opponent Actions")
        plt.ylabel("Agent Actions")
        plt.show()

# Plot heatmaps for both JAL agents
plot_q_values_heatmap(agent_0_jal, "Agent 0 (JAL)")
plot_q_values_heatmap(agent_1_jal, "Agent 1 (JAL)")


# In[13]:


# Plot heatmaps for IL agents:
def plot_q_values_heatmap_il(agent, agent_name):
    for state in range(agent.state_size):
        plt.figure(figsize=(8, 6))
        q_values = agent.qtable[state]  # Get the Q-values for this state
        sns.heatmap(q_values.reshape(-1, 1), annot=True, cmap="coolwarm", cbar=True,
                    yticklabels=[f"Action {i}" for i in range(agent.action_size)],
                    xticklabels=[f"State {state}"])
        plt.title(f"Q-Values for {agent_name} in State {state}")
        plt.xlabel("State")
        plt.ylabel("Agent Actions")
        plt.show()

plot_q_values_heatmap_il(agent_0_il, "Agent 0 (IL)")
plot_q_values_heatmap_il(agent_1_il, "Agent 1 (IL)")


# In[16]:

"""

"""
Build a matrix for the stochastic game with a 3x3x3 game matrix
"""
# Define the means and standard deviations from the table
# Each entry is (mean, standard_deviation)

from StochasticGame import StochasticGame, StandardDeviation

STOCHASTIC_GAME_REWARDS = np.array([
    # Player 1: a1
    [[11, -30, 0],     # Player 2: b1, for Player 3: c1, c2, c3
     [-30, 7, 6],      # Player 2: b2, for Player 3: c1, c2, c3
     [0, 0, 5]],         # Player 2: b3, for Player 3: c1, c2, c3
    
    # Player 1: a2
    [[11, -3, 0],      # Player 2: b1, for Player 3: c1, c2, c3
     [-3, 7, 6],       # Player 2: b2, for Player 3: c1, c2, c3
     [0, 0, 3]],         # Player 2: b3, for Player 3: c1, c2, c3

    # Player 1: a3
    [[14, 0, 0],       # Player 2: b1, for Player 3: c1, c2, c3
     [-50, 4, 3],      # Player 2: b2, for Player 3: c1, c2, c3
     [0, 0, 3]]          # Player 2: b3, for Player 3: c1, c2, c3
])


"""
Implement Independent Learners, and several types of Joint-Action learners in the above problem.
The first Joint-Action learner with simple Boltzmann action selection.
Make sure the temperature decays slowly to eventually reach a fully exploiting policy.

You can find Boltzmann action selection in the paper
of Claus & Boutilier [1] Then implement the more advanced action selection strategies that you find in the
Section 5 of Claus & Boutilier’s paper, i.e., Optimistic Boltzmann, Weighted Optimistic Boltzmann, and
the combination of both.  Recall that in Joint-Action learning, you learn about the quality of joint actions(27 in total in this setting), 
and you often have to maintain beliefs about the other agents’ strategies.    
"""

from JointActionLearner import BoltzmannJointActionLearner

# Number of trial runs
num_trials = 2
num_episodes = 10
# Store the probability of optimal joint action for each episode, averaged across trials
optimal_action_probs_boltzmann = np.zeros(num_episodes)

optimal_joint_actions = [(0,0,0), (1,0,0), (2, 0, 0)] # I think!

# Run trial runs for Boltzmann Joint Action Learner
for trial in range(num_trials):

    # Initialize environment and agents for Joint Action Learners (JALs)
    standard_deviations = StandardDeviation(sigma0=2, sigma1=2, sigma=2)

    env = StochasticGame(reward_matrix=STOCHASTIC_GAME_REWARDS, standard_deviations=standard_deviations)

    agent_0_boltzmann = BoltzmannJointActionLearner(
        action_size=3,
        opponent_action_size=3,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        temperature=1.0,
        temperature_min=0.01,
        temperature_decay=0.95
    )

    agent_1_boltzmann = BoltzmannJointActionLearner(
        action_size=3,
        opponent_action_size=3,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        temperature=1.0,
        temperature_min=0.01,
        temperature_decay=0.95
    )

    agent_2_boltzmann = BoltzmannJointActionLearner(
        action_size=3,
        opponent_action_size=3,
        state_size=1,
        learning_rate=0.0,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        temperature=1.0,
        temperature_min=0.01,
        temperature_decay=0.95
    )

    # Keep track of the number of optimal actions for each episode
    optimal_actions_boltzmann = np.zeros(num_episodes)

    for episode in range(num_episodes):

        observations = env.reset()
        state = 0

        done = False

        # Agents take actions
        action_0_boltzmann = agent_0_boltzmann.select_action(state)
        action_1_boltzmann = agent_1_boltzmann.select_action(state)
        action_2_boltzmann = agent_2_boltzmann.select_action(state)

        actions = {"player_0": action_0_boltzmann, "player_1": action_1_boltzmann, "player_2": action_2_boltzmann}

        # Step in the environment
        observations, rewards, dones, _, infos = env.step(actions)

        # Check if the agents selected the optimal joint action
        #print("actions", action_0_boltzmann, action_1_boltzmann, action_2_boltzmann)
        if (action_0_boltzmann, action_1_boltzmann, action_2_boltzmann) in optimal_joint_actions:
            print("yes")
            optimal_actions_boltzmann[episode] += 1

        # Update the Q-tables
        agent_0_boltzmann.update(state, action_0_boltzmann, action_1_boltzmann, action_2_boltzmann, state, rewards["player_0"], dones["player_0"])
        agent_1_boltzmann.update(state, action_1_boltzmann, action_0_boltzmann, action_2_boltzmann, state, rewards["player_1"], dones["player_1"])
        agent_2_boltzmann.update(state, action_2_boltzmann, action_0_boltzmann, action_1_boltzmann, state, rewards["player_2"], dones["player_2"])
        """
        
        # Update the Q-tables
        agent_0_jal.update(state, action_0_jal, action_1_jal, state, rewards["player_0"], dones["player_0"])
        agent_1_jal.update(state, action_1_jal, action_0_jal, state, rewards["player_1"], dones["player_1"])
        """

        # Update belief about the opponent's strategy
        agent_0_boltzmann.update_opponent_distribution(action_1_boltzmann, action_2_boltzmann)
        agent_1_boltzmann.update_opponent_distribution(action_0_boltzmann, action_2_boltzmann)
        agent_2_boltzmann.update_opponent_distribution(action_0_boltzmann, action_1_boltzmann)

        # Update temperature
        agent_0_boltzmann.update_temperature()
        agent_1_boltzmann.update_temperature()
        agent_2_boltzmann.update_temperature()

    # Calculate the probability of optimal joint action over episodes for this trial
    optimal_action_probs_boltzmann += optimal_actions_boltzmann / num_trials  # Averaging across trials
    print("optimal_actions_boltzmann/num_trials", optimal_actions_boltzmann / num_trials )

# Plot the results with JAL for 3 agents
plt.plot(optimal_action_probs_boltzmann, label='Boltzmann Joint Action Learners')
plt.xlabel('Number of Interactions')
plt.ylabel('Probability of Choosing an Optimal Joint Action')
plt.title('Choosing Optimal Joint Actions')
plt.legend()
plt.grid(True)
plt.show()
