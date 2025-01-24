from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from stable_baselines3.common.env_checker import check_env

from dataclasses import dataclass

from utils.conductance_data import conductanceData


@dataclass
class SliderLimits:
    """
    Container class for figure slider limits
    """
    zeeman_min = 0.0
    zeeman_max = 0.3
    potential_min = 0
    potential_max = 1.5
    modulation_min = -1.5
    modulation_max = 1.5


class ConductanceEnv(gym.Env):
    """
    Observation space:
        dIdV at weak and strong tunneling (200,) array
    Actions (Continuous):
        0: Change the chemical potential
        1: Change the Zeeman field
        2: Change the modulation strength
    Reward:
        Change in state value. See get_value
    Params:
        time_penalty: The time penalty for each step
        render_mode: The render mode, either None, "human" or "rgb_array"
        obs_size: The size of the observation
        n: The number of unit cells in the supercell
        noise: The noise level of the observation
    """
    metadata = {'render_modes': [None]}

    def __init__(self, random_range=0.0, time_penalty=0.0, render_mode=None, noise=0.0, limit_actions=False):
        super().__init__()
        assert render_mode is None, "Rendering is not supported"
        self.render_mode = None  # Fix the SB3 vec env breaking if this attribute doesn't exist

        self.render_mode = render_mode
        self.slider_limits = SliderLimits()  # For webgui sliders

        self.obs_size = obs_size = 200
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        self.random_range = random_range
        self.time_penalty = time_penalty
        self.noise = noise
        self.limit_actions = limit_actions
        self.__potential = 0.0
        self.__zeeman = 0.0
        self.__modulation = 0.0  # Hidden versions to use if limit_actions is True

        self.data = conductanceData
        conductanceData.load_data()
        self.potential = 0.0
        self.zeeman = 0.0
        self.modulation = 0.0
        self.__old_value = 0.0
        self.__berry = 0.0
        self.__gap = 0.0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.__init_hamiltonian()
        # Reset the initial state until we get a trivial state
        while value := self.get_value() > 0.0:
            self.__init_hamiltonian()

        obs = self.get_obs()
        self.__old_value = value

        return obs, {"chern": self.__berry, "gap": self.__gap}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.scale_action(action)

        if self.__check_limits(action):
            return self.get_obs(), -0.5, True, False, {"chern": self.__berry, "gap": self.__gap}

        self.apply_action(action)

        obs = self.get_obs()
        # Reward is the change in the state value
        new_value = self.get_value()
        reward = new_value - self.__old_value
        self.__old_value = new_value

        terminated = bool(new_value > 0.25)
        truncated = False
        info = {"chern": self.__berry, "gap": self.__gap}
        reward -= self.time_penalty

        return obs, reward, terminated, truncated, info

    def __check_limits(self, action: ActType) -> bool:
        """
        Checks if applying the action stays within physical limits. Returns True if the action goes out of bounds.
        """
        if self.limit_actions:
            return False
        potential_check = 0 <= self.potential + action[0] <= 1.5
        zeeman_check = 0 <= self.zeeman + action[1] <= 0.3
        return not all([potential_check, zeeman_check])

    @staticmethod
    def soft_step(x: float, start: float, stop: float, min: float, max: float) -> float:
        """
        Smoothly restricts x to min and max outside the interval [start, stop] using tanh
        """
        slope = 3
        return min + (max - min) * 0.5 * (np.tanh((x - start / 2 - stop / 2) / (stop - start) * slope) * 1.0 + 1)

    def get_value(self) -> float:
        """
        Returns the value of the current state.
        The value is defined as |berry| * gap
        """
        berry = self.get_invariant()
        gap = self.get_gap()
        return berry * gap

    def get_obs(self) -> np.ndarray:
        """
        Returns the observation. Left half of the observation is weak didv, right half is strong didv.
        Applies noise renormalization followed by normalization to [0, 1]
        """
        obs = self.data.get_observation(self.potential, self.zeeman, self.modulation)
        noise = self.np_random.normal(0, self.noise, obs.shape)
        obs = (1 + noise) * obs
        obs = obs / (np.max(obs) + 1e-9)
        return obs.astype(np.float32)

    def get_gap(self) -> float:
        berry, gap = self.data.get_berry(self.potential, self.zeeman, self.modulation)
        berry = np.abs(berry)
        self.__berry = berry
        self.__gap = gap
        return gap

    def get_invariant(self) -> float:
        berry, gap = self.data.get_berry(self.potential, self.zeeman, self.modulation)
        berry = np.abs(berry)
        self.__berry = berry
        self.__gap = gap
        return berry

    @staticmethod
    def scale_action(action: ActType) -> ActType:
        """
        Scales the action to give the agent finer control over the parameters.
        """
        action[0] *= 1.0
        action[1] *= 0.2
        action[2] *= 1.0
        return action

    def apply_action(self, action: ActType) -> None:
        """
        Applies the action to the environment
        """
        if not self.limit_actions:
            self.potential += action[0]
            self.zeeman += action[1]
            self.modulation += action[2]
        else:
            self.__potential += action[0]
            self.__zeeman += action[1]
            self.__modulation += action[2]
            self.potential = self.soft_step(self.__potential, 0.0, 1.2, 0.0, 1.2)
            self.zeeman = self.soft_step(self.__zeeman, 0.0, 0.3, 0.0, 0.3)
            self.modulation = self.soft_step(self.__modulation, -1.5, 1.5, -1.5, 1.5)
            # self.potential = min(max(self.__potential, 0.0), 1.5)
            # self.zeeman = min(max(self.__zeeman, 0.0), 0.3)
            # self.modulation = min(max(self.__modulation, -1.5), 1.5)

    def set_params(self, values: ActType) -> None:
        """
        Sets environment parameters to the given values
        """
        self.potential = self.__potential = values[0]
        self.zeeman = self.__zeeman = values[1]
        self.modulation = self.__modulation = values[2]

    def __random_factor(self) -> float:
        """
        Samples the uniform distribution [1 - random_range, 1 + random_range]
        """
        low = 1 - self.random_range
        high = 1 + self.random_range
        return self.np_random.uniform(low, high)

    def __random_between(self, low: float, high: float) -> float:
        """
        Returns a random number from the uniform distribution [low, high]
        """
        return self.np_random.uniform(low, high)

    def __init_hamiltonian(self) -> None:
        """
        Initializes the Hamiltonian parameters to a random state
        """
        modulation = self.__random_between(0, 1.5)
        potential = self.__random_between(0, 1.5)
        zeeman = self.__random_between(0.0, 0.3)
        self.set_params([potential, zeeman, modulation])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Plots the Differential Conductance
        Returns:
            RGBA Array (H, W, 4)
        Not implemented yet
        """
        raise NotImplementedError


if __name__ == '__main__':
    env = ConductanceEnv()
    print("Env dimensions")
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.sample().shape)
    print("Checking environment")
    check_env(env, warn=True)
    print("Check successful, starting benchmark")
    from time import perf_counter

    start = perf_counter()
    for i in range(1000):
        env.reset()
        env.step(env.action_space.sample())
    print(f"Env runs at: {1000 / (perf_counter() - start):.2e} fps")
