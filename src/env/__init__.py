import gymnasium as gym
from gymnasium.envs.registration import register
from lunar_lander import LunarLander

register(
    id='env/LunarLander-v0',
    entry_point='env:LunarLander',
    max_episode_steps=1000,
)