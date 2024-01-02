import gymnasium as gym
import numpy as np
from agent import Agent
from collections import deque
import env

N_EPISODES = 20

env = gym.make('env/LunarLander-v0', render_mode="human")
agent = Agent()
agent.load("agent-1.pth")

scores = []
fuels = []

for _ in range(N_EPISODES):
	state = env.reset()[0]
	score = 0
	fuel = 0
	angle = []
	angle.append(abs(state[4]))

	while(True):
		action = agent.get_action(state)
		new_state, reward, terminated, truncated, info = env.step(action)
		state = new_state
		angle.append(abs(new_state[4]))
		if action == 1 | action == 3:
			fuel += 1
		elif action == 2:
			fuel += 1
		score += reward
		if terminated or truncated:
			landing_accuracy = np.sqrt(state[0] * state[0] + state[1] * state[1])
			print("Episode {} \tReward: {:.2f}\tFuel: {:.1f}\tAngle: {:.2f}\tLanding accuracy: {:.2f}".format(_, score, fuel, np.mean(angle), landing_accuracy))
			break