import gymnasium as gym
import numpy as np

class GymWrapper:
    def __init__(self, env_name="CartPole-v1", noise_std=0.0):
        self.env = gym.make(env_name)
        self.noise_std = noise_std
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()
        return self._noise(state)

    def step(self, action):
        state, reward, terminated, truncated, _ = self.env.step(action)
        return self._noise(state), reward, terminated or truncated

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _noise(self, state):
        if self.noise_std > 0:
            state += np.random.normal(0, self.noise_std, size=state.shape)
        return state
