"""
Frozen Lake example using OpenAI Gym.
"""

### IMPORT MODULES ###

import numpy as np
from numpy import savetxt
import gym
import random
import time
from IPython.display import clear_output


### CREATE AN ENVIRONEMNT ###

# Check the available environments on the gym website
env = gym.make("FrozenLake-v1")


### CONSTRUCT Q_TABLE ###

# Initialize all q-values to zero for each state-action pair
# Remember, the number of rows in the table is equivalent to
# the size of the state space in the environment, and the number
# of columns is equivalent to the size of the action space.

# Get info from the environment
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Build q-table and fill it with zeros
# Columns => Actions, Rows => States
q_table = np.zeros((state_space_size, action_space_size))
print(q_table)


### INITIALIZE PARAMETERS ##

# Create and initialize all the parameters required for q-learning
# algorithm.

num_episodes = 10000	
max_steps_per_episode = 100  # Terminate episode no matter what the outcome or current state is

learning_rate = 0.1
discount_rate = 0.99

# Exploration-exploitation trade-off parameters. in regards
# to epsilon greedy strategy
exploration_rate = 1 # epsilon
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


### Q-LEARNING ALGORITHM ###
rewards_all_episodes = []

for episode in range(num_episodes):  # Contains everything that happens in a single episode
	state = env.reset()

	done = False
	rewards_current_episode = 0

	for step in range(max_steps_per_episode):  # Contains everything that happens in one step within each episode

		# Exploration-exploitation trade-off
		exploration_rate_threshold = random.uniform(0, 1)
		if exploration_rate_threshold > exploration_rate:  # Agent will exploit the enmvironment
			action = np.argmax(q_table[state, :])
		else:
			action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)  # We take the choosen action with env.step()

		# Update Q-table for Q(s, a)
		q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
			learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

		state = new_state
		rewards_current_episode += reward

		# Check if last action ended the episode for the agent i.e.
		# agent stepped into a hole or reached the goal. If the 
		# episdoe end, then we jump out of the neested loop and 
		# move to the new episode. If not then we move on to the 
		# next step within same episode.
		if done == True:
			break

	# Exploration rate decay
	# After the episode finished, we need to update our exploration rate
	exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
		np.exp(-exploration_decay_rate * episode)

	# Append teh reward
	rewards_all_episodes.append(rewards_current_episode)


### CHECKING OUTPUTS ###

# Calculate and print the average reward per thousand episodes
reward_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("******Average reward per thousand episodes******\n")
for r in reward_per_thousand_episodes:
	print(count, ": ", str(sum(r/1000)))
	count += 1000

# Print updated Q-table
print("\n\n******Q=table******\n")
print(q_table)
print(q_table.dtype)

# Save the Q-table for inference
savetxt('q-table.txt', q_table, delimiter=' ')