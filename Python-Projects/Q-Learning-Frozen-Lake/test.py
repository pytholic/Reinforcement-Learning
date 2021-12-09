"""
Frozen Lake example using OpenAI Gym.
"""

### IMPORT MODULES ###

import numpy as np
from numpy import loadtxt
import gym
import random
import time
from IPython.display import clear_output  # For notebook


num_episodes = 3
max_steps_per_episode = 100 

# Create environment
env = gym.make("FrozenLake-v1")

# Load the saved Q-table
q_table = loadtxt('q-table.txt', delimiter=' ')
#print(q_table)


### AGENT PLAYS FROZEN LAKE ###

for episode in range(num_episodes):
	state = env.reset()
	done = False
	print("******EPISODE ", episode + 1, "******\n\n\n\n")
	time.sleep(1)

	for step in range(max_steps_per_episode):
		clear_output(wait=True)
		env.render()
		time.sleep(0.3)

		action = np.argmax(q_table[state, :])
		new_state, reward, done, info = env.step(action)

		if done:
			clear_output(wait=True)
			env.render()
			if reward == 1:
				print("****You reached the goal!****")
				time.sleep(3)
			else:
				print("****You fell through a hole!****")
				time.sleep(3)
			clear_output(wait=True)
			break

		state = new_state

env.close()