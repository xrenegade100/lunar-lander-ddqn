import gymnasium as gym
import numpy as np
from agent import Agent
from collections import deque
import env

N_EPISODES = 50

env = gym.make('env/LunarLander-v0', render_mode="human")
agent = Agent()
agent.load("agent-1.pth")

reward_window = deque(maxlen=100)
fuel_window = deque(maxlen=100)

for _ in range(N_EPISODES):
	state = env.reset()[0]
	score = 0
	fuel = 0

	while(True):
		action = agent.get_action(state)
		new_state, reward, terminated, truncated, info = env.step(action)
		state = new_state
		if action == 1 | action == 3:
			fuel += 0.1
		elif action == 2:
			fuel += 0.5
		score += reward
		if terminated or truncated:
			print("Episode {} \t Reward: {}\t Fuel: {}".format(_, score, fuel))
			break

		fuel_window.append(fuel)

print("Mean Reward: {}".format(np.mean(reward_window)))
print("Mean Fuel: {}".format(np.mean(fuel_window)))