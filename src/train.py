import gymnasium as gym
import numpy as np
from agent import Agent
from collections import deque
import env

N_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
EPS = 1.0

env = gym.make('env/LunarLander-v0')
agent = Agent(eps=EPS, lr=5e-3, batch_size=128)

scores = []
scores_window = deque(maxlen=100)
fuel_window = deque(maxlen=100)

for _ in range(N_EPISODES):
	state = env.reset()[0]
	episode_score = 0
	fuel = 0

	for t in range(MAX_STEPS_PER_EPISODE):
		action = agent.get_action(state)
		new_state, reward, terminated, truncated, info = env.step(action)
		if action == 1 | action == 3:
			fuel += 0.1
		elif action == 2:
			fuel += 0.5
		agent.save_transition(state, action, reward, new_state, terminated)
		agent.learn()
		state = new_state

		episode_score += reward

		if terminated or truncated:
			break

	scores_window.append(episode_score)
	fuel_window.append(fuel)

	print('\rEpisode {}\tAverage Score: {:.2f}\tAvg Loss: {:.2f}\tFuel: {:.2f}'.format(_, np.mean(scores_window), np.mean(agent.loss_window), np.mean(fuel_window)), end="")
	if _ % 100 == 0:
		print('\rEpisode {}\tAverage Score: {:.2f}\tAvg Loss: {:.2f}\tFuel: {:.2f}'.format(_, np.mean(scores_window), np.mean(agent.loss_window), np.mean(fuel_window)))

agent.save("agent-1.pth")