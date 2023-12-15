import gymnasium as gym
import numpy as np
from agent import Agent
from collections import deque

N_EPISODES = 50

env = gym.make('LunarLander-v2', render_mode="human")
agent = Agent()
agent.load("agent-1.pth")

for _ in range(N_EPISODES):
	state = env.reset()[0]
	score = 0

	while(True):
		action = agent.get_action(state)
		new_state, reward, terminated, truncated, info = env.step(action)
		state = new_state
		score += reward
		if terminated or truncated:
			print("Episode {} \t Reward: {}".format(_, score))
			break