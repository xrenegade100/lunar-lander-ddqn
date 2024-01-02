from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BufferMemory:

	def __init__(self, max_size):
		self.__buffer = deque(maxlen=max_size)
		self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])
	
	def save(self, state, action, reward, new_state, terminal):
		t = self.transition(state, action, reward, new_state, terminal)
		self.__buffer.append(t)

	def get_counter(self):
		return len(self.__buffer)
	
	def get_batch(self, batch_size):
		experiences = random.sample(self.__buffer, k=batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.new_state for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.terminal for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)

class DQNetwork(nn.Module):

	STATE_SIZE = 8
	ACTION_SIZE = 4

	def __init__(self):
		super(DQNetwork, self).__init__()
		self.fc1 = nn.Linear(self.STATE_SIZE, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, self.ACTION_SIZE)

	def forward(self, state):
		x = self.fc1(state)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		return self.fc3(x)
	
class Agent:

	def __init__(self, lr = 1e-3, memory_max_size = int(1e5), learn_every = 4, batch_size = 64, gamma = 0.99, tau = 0.001, eps = 0, eps_dec = 0.995, eps_min = 0.01):
		self.__qnetwork = DQNetwork().to(device)
		self.__qnetwork_target = DQNetwork().to(device)
		self.__optimizer = optim.Adam(self.__qnetwork.parameters(), lr=lr)
		self.__memory = BufferMemory(memory_max_size)
		self.__learn_every = learn_every
		self.__step = 0
		self.__batch_size = batch_size
		self.__gamma = gamma
		self.__tau = tau
		self.__eps = eps
		self.__eps_dec = eps_dec
		self.__eps_end = eps_min
		self.loss_window = deque(maxlen=100)
	
	def load(self, path):
		self.__qnetwork.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.__qnetwork.state_dict(), path)

	def save_transition(self, state, action, reward, new_state, terminal):
		self.__memory.save(state, action, reward, new_state, terminal)
	
	def learn(self):
		self.__step = (self.__step + 1) % self.__learn_every

		if self.__step == 0 and self.__memory.get_counter() >= self.__batch_size:
			transitions = self.__memory.get_batch(self.__batch_size)
			self.__learn(transitions)

	def __learn(self, transitions):
		self.__qnetwork.train()
		states, actions, rewards, next_states, terminals = transitions

		# Get max estimated Q values (for next states) from target network
		# and detach the resulting tensor from the computational graph
		q_targets_next = self.__qnetwork_target(next_states).detach()

		# Bellman equation to update weights
		q_values = rewards + self.__gamma * q_targets_next.max(1)[0].unsqueeze(1) * (1 - terminals)

		q = self.__qnetwork(states).gather(1, actions)

		loss = F.mse_loss(q, q_values)
		self.__optimizer.zero_grad()
		loss.backward()
		self.__optimizer.step()
		self.__soft_update()

		loss_val = loss.cpu().data.numpy()
		self.loss_window.append(loss_val)

	def __soft_update(self):
		for target_param, local_param in zip(self.__qnetwork_target.parameters(), self.__qnetwork.parameters()):
			target_param.data.copy_(self.__tau * local_param.data + (1.0-self.__tau) * target_param.data)
	
	def get_action(self, state):
		rand = random.uniform(0, 1)
		# print(rand, self.__eps)
		if rand < self.__eps:
			# Reduce eps
			if self.__eps > self.__eps_end:
				self.__eps = self.__eps * self.__eps_dec
			
			return np.random.choice([i for i in range(4)])
		
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.__qnetwork.eval()
		with torch.no_grad():
			action = self.__qnetwork(state)
		
		# Reduce eps
		if self.__eps > self.__eps_end:
			self.__eps = self.__eps * self.__eps_dec

		return np.argmax(action.cpu().data.numpy())
