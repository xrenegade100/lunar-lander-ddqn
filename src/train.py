import gymnasium as gym
import numpy as np
from agent import Agent
from collections import deque
import env
import matplotlib.pyplot as plt

N_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1000
EPS = 1.0

def plot_convergence(scores, fuel_consumption, landings, angles):
    plt.figure(figsize=(18, 8))

    plt.subplot(4, 1, 1)
    plt.plot(scores)
    plt.title('Andamento delle Ricompense nel Tempo')
    plt.xlabel('Episodi x1000')
    plt.ylabel('Ricompense')
    
    plt.subplot(4, 1, 2)
    plt.plot(fuel_consumption)
    plt.title('Consumo di Carburante nel Tempo')
    plt.xlabel('Episodi x1000')
    plt.ylabel('Carburante')

    plt.subplot(4, 1, 3)
    plt.plot(landings)
    plt.title('Andamento della precisione di atteraggio nel Tempo')
    plt.xlabel('Episodi x1000')
    plt.ylabel('Precisione atteraggi')
    
    plt.subplot(4, 1, 4)
    plt.plot(angles)
    plt.title('Andamento della stabilità nel Tempo')
    plt.xlabel('Episodi x1000')
    plt.ylabel('Stablità lunar-lander')

    plt.tight_layout()
    plt.show()


# Custom env
env = gym.make('env/LunarLander-v0')

# Normal env
# env = gym.make('LunarLander-v2')

agent = Agent(eps=EPS, lr=5e-4, batch_size=128)

scores = []
fuels = []
landings = []
angles = []
angle_window = deque(maxlen=100)
scores_window = deque(maxlen=100)
fuel_window = deque(maxlen=100)
landings_window = deque(maxlen=100)

for _ in range(N_EPISODES):
	state = env.reset()[0]
	episode_score = 0
	fuel = 0
	angle = []
	angle.append(abs(state[4]))

	for t in range(MAX_STEPS_PER_EPISODE):
		action = agent.get_action(state)
		new_state, reward, terminated, truncated, info = env.step(action)

		angle.append(abs(new_state[4]))
		if action == 1 | action == 3:
			fuel += 1
		elif action == 2:
			fuel += 1
		agent.save_transition(state, action, reward, new_state, terminated)
		agent.learn()
		state = new_state

		episode_score += reward

		if terminated or truncated:
			break

	# Euclidean distance from the origin
	landing_accuracy = np.sqrt(state[0] * state[0] + state[1] * state[1])
	
	landings_window.append(landing_accuracy)
	angle_window.append(np.mean(angle))
	scores_window.append(episode_score)
	fuel_window.append(fuel)

	print('\rEpisode {}\tAverage Score: {:.2f}\tAvg Loss: {:.2f}\tFuel: {:.2f}\tAngle: {:.2f}\tLandings accuracy: {:.2f}'.format(_, np.mean(scores_window), np.mean(agent.loss_window), np.mean(fuel_window), np.mean(angle_window), np.mean(landings_window)), end="")
	if _ % 100 == 0:
		print('\rEpisode {}\tAverage Score: {:.2f}\tAvg Loss: {:.2f}\tFuel: {:.2f}\tAngle: {:.2f}\tLandings accuracy: {:.2f}'.format(_, np.mean(scores_window), np.mean(agent.loss_window), np.mean(fuel_window), np.mean(angle_window), np.mean(landings_window)))
		scores.append(np.mean(scores_window))
		fuels.append(np.mean(fuel_window))
		landings.append(np.mean(landings_window))
		angles.append(np.mean(angle_window))

agent.save("agent-1.pth")

# Chiama la funzione per plottare i grafici
plot_convergence(scores, fuels, landings, angles)