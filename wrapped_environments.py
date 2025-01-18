from env_wrapper import Wrapper, Action_Space
from fasttttsandbox import TTTNvN
import numpy as np
import gym
import time
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


class CartpoleWrapped(Wrapper):
    def __init__(self, render=None):
        self.env = gym.make("CartPole-v1", render_mode=render)
        self.state = np.array([0, 0, 0, 0])
        self.n_agents = 1
        self.n_actions = 2
        self.action_space = Action_Space(2)
        self.action = 0

    def reset(self):
        self.state, info = self.env.reset()
        return np.array([self.state]), info

    def get_state_feature_names(self):
        return ["pos", "vel", "angle", "angular vel"]

    def get_obs_feature_names(self):
        return ["pos", "vel", "angle", "angular vel"]

    def get_obs(self):
        return [self.state]

    def get_avail_agent_actions(self, agent_id):
        return None

    def step(self, actions):
        next_state, reward, terminated, truncated, info = self.env.step(actions[0])

        self.state = next_state
        return np.array([self.state]), reward, terminated, truncated, info

    def expert_reward(self, obs):
        return abs(obs[0][3] / 10)

    def display(self, obs, avail, id):
        self.env.render()
        time.sleep(0.1)

    def human_action(self, obs, avail_actions, agent_id, keys_down):
        if "a" in keys_down and keys_down["a"]:
            self.action = 0
        elif "d" in keys_down and keys_down["d"]:
            self.action = 1
        else:
            self.action = int(not bool(self.action))
        return np.array([self.action])


class OvercookWrapped(Wrapper):
    def __init__(self, render=None, max_timestep=200):
        self.max_timestep = max_timestep
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(mdp, horizon=max_timestep)
        print(self.env)
        # {
        #    "both_agent_obs": both_agents_ob,
        #    "overcooked_state": self.base_env.state,
        #    "other_agent_env_idx": 1 - self.agent_idx,
        # }
        # self.env.lossless_state_encoding_mdp()
        self.env.reset()
        st: OvercookedState = self.env.state
        self.info = st.to_dict()
        self.obs = self.state_vecorizer(st)
        self.n_agents = 2
        self.n_actions = 6
        self.action_space = Action_Space(6)
        self.action = 0

    def state_vecorizer(self, state):
        self.info = state.to_dict()
        enc = self.env.lossless_state_encoding_mdp(state)

        obs = np.array([enc[0].flatten(), enc[1].flatten()])
        return obs

    def reset(self):
        self.env.reset()
        st: OvercookedState = self.env.state
        self.info = st.to_dict()
        self.obs = self.state_vecorizer(st)
        return self.obs, self.info

    def get_state_feature_names(self):
        return self.env.state.to_dict().keys()

    def get_obs_feature_names(self):
        return self.env.state.to_dict().keys()

    def get_obs(self):
        obs = self.state_vecorizer(self.env.state)
        return obs

    def get_avail_agent_actions(self, agent_id):
        return np.ones(6)

    def step(self, actions):
        action_set = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), "interact"]
        b = [action_set[actions[0]], action_set[actions[1]]]

        next_obs, reward, terminated, info = self.env.step(b)
        self.obs = next_obs
        return self.state_vecorizer(next_obs), reward, terminated, False, info

    def expert_reward(self, obs):
        return abs(obs[0][3] / 10)

    def display(self, obs, avail, id):
        print(self.env)
        print(self.env.state.to_dict()["timestep"])
        time.sleep(0.1)

    def human_action(self, obs, avail_actions, agent_id, keys_down):
        action = input(f"action for agent [{agent_id}]: ")
        if action == "w":
            self.action = 0
        elif action == "s":
            self.action = 1
        elif action == "d":
            self.action = 2
        elif action == "a":
            self.action = 3
        elif action == "z":
            self.action = 5
        else:
            self.action = 4
        return np.array([self.action])


class TTTWrapped(Wrapper):
    def __init__(
        self, nfirst=1, n_moves=1, render_mode="", random_op=True, obs_as_array=True
    ):
        self.env = TTTNvN(
            nfirst=nfirst,
            n_moves=n_moves,
            render_mode=render_mode,
            random_op=random_op,
            obs_as_array=obs_as_array,
        )
        self.state = np.zeros(18, dtype=np.float32)
        self.n_agents = n_moves
        self.n_actions = 9
        self.action_space = Action_Space(9)
        self.action = 0

    def reset(self):
        self.state, info = self.env.reset()
        return np.array([self.state] * self.env.n_moves), info

    # TODO: make add agent id to the states for role assignment

    def get_state_feature_names(self):
        return ["1,1", "1,2", "1,3", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3"]

    def get_obs_feature_names(self):
        return ["1,1", "1,2", "1,3", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3"]

    def get_obs(self):
        return [self.state] * self.env.n_moves

    def get_avail_agent_actions(self, agent_id):
        return None  # TODO: eventually make this return env.get_legal_moves

    def step(self, actions):
        next_state, reward, terminated, truncated, info = self.env.step(actions)

        self.state = next_state
        return (
            np.array([self.state] * self.env.n_moves),
            reward,
            terminated,
            truncated,
            info,
        )

    def expert_reward(self, obs):
        return 0

    def display(self, obs, avail, id):
        self.env.display_board(self.env.board)

    def human_action(self, obs, avail_actions, agent_id, keys_down):
        action = int(input(f"action for agent [{agent_id}] [1-9]: "))
        self.action = action
        return np.array([self.action])
