from abc import ABC, abstractmethod
import numpy as np
import time


class Action_Space:
    def __init__(self, size):
        self.size = size
        self.n = size

    def sample(self):
        return np.random.randint(0, self.size)


class Wrapper(ABC):
    def __init__(self, render=None):
        return

    @abstractmethod
    def reset(self):
        info = 0
        self.state = 0
        return [self.state], info

    @abstractmethod
    def get_state_feature_names(self):
        return ["pos", "vel", "angle", "angular vel"]

    @abstractmethod
    def get_obs_feature_names(self):
        return ["pos", "vel", "angle", "angular vel"]

    @abstractmethod
    def get_obs(self):
        return [self.state]

    @abstractmethod
    def get_avail_agent_actions(self, agent_id):
        return np.ones(2)

    @abstractmethod
    def step(self, actions):
        next_state, reward, terminated, truncated, info = (
            0,
            0,
            0,
            0,
            0,
        )  # self.env.step(actions[0])

        return self.state, reward, terminated, truncated, info

    @abstractmethod
    def expert_reward(self, obs):
        return abs(obs[0][3] / 10)

    @abstractmethod
    def display(self, obs, avail, id, human=False):
        print("abstract render")

    @abstractmethod
    def human_action(self, obs, avail_actions, agent_id, keys_down):
        return 0
