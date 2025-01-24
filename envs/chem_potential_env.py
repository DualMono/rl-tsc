from io import BytesIO
from typing import Any, SupportsFloat
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

from pyqula import geometry
from pyqula.hamiltonians import Hamiltonian

from stable_baselines3.common.env_checker import check_env


class ChemPotentialEnv(gym.Env):
    """
    Observation space:
        Density of states: (100,) array
    Actions (Discrete):
        0: Stop if time_penalty is > 0, else nothing
        1: Decrease the chemical potential
        2: Increase the chemical potential
    Actions (Continuous):
        0: Stop if time_penalty is > 0 and action > 0.5, else nothing
        1: Change the chemical potential
    Reward:
        Change in |C|*gap
    """
    metadata = {'render.modes': "rgb_array", "render_fps": 2}

    def __init__(self, random_range=0.1, time_penalty=0.0001, continuous=True, render_mode=None,
                 random_lattice=False):
        super().__init__()
        self.render_mode = render_mode
        self.random_lattice = random_lattice

        self.continuous = continuous
        if not continuous:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(100,), dtype=np.float32)

        self.random_range = random_range
        self.time_penalty = time_penalty

        self.h: Hamiltonian | None = None
        self.potential = 0.0
        self.zeeman = 0.0
        self.__old_value = 0.0
        self.__chern = 0.0
        self.__gap = 0.0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.h = self.__init_hamiltonian()
        self.potential = 0

        obs = self.get_obs()
        self.__old_value = self.get_value()
        return obs, {"chern": self.__chern, "gap": self.__gap}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.h is not None, "You forgot to call reset()"
        if not self.continuous:
            if action == 0 and self.time_penalty > 0:
                # Agent is done
                obs = self.get_obs()
                reward = 0
                terminated = True
                truncated = False
                info = {"chern": self.__chern, "gap": self.__gap}
                return obs, reward, terminated, truncated, info
            if action == 1:
                self.h.add_onsite(-0.1)
                self.potential -= 0.1
            elif action == 2:
                self.h.add_onsite(0.1)
                self.potential += 0.1
        else:
            if action[0] > 0.5 and self.time_penalty > 0:
                # Agent is done
                obs = self.get_obs()
                reward = 0
                terminated = True
                truncated = False
                info = {"chern": self.__chern, "gap": self.__gap}
                return obs, reward, terminated, truncated, info
            else:
                self.h.add_onsite(action[1])
                self.potential += action[1]
        obs = self.get_obs()
        # Reward is the change in the chern number
        new_value = self.get_value()
        reward = new_value - self.__old_value
        self.__old_value = new_value
        terminated = False
        truncated = False
        info = {"chern": self.__chern, "gap": self.__gap}
        # Penalize taking long if early stopping is an option
        reward -= self.time_penalty
        return obs, reward, terminated, truncated, info

    def get_value(self) -> float:
        """
        Returns the value of the current state. The reward is defined as |C|*gap
        """
        self.__chern = np.abs(self.h.get_chern())
        self.__gap = self.h.get_gap()
        return np.abs(self.__chern * self.h.get_gap())

    def get_obs(self, return_energy=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Returns the density of states as a (100,) array
        """
        (e, dos) = self.h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50)
        dos: np.ndarray = dos.astype(np.float32)
        if return_energy:
            return e, dos
        else:
            return dos

    def __random_factor(self) -> float:
        """
        Samples the uniform distribution [1 - random_range, 1 + random_range]
        """
        low = 1 - self.random_range
        high = 1 + self.random_range
        return self.np_random.uniform(low, high)

    def __init_hamiltonian(self) -> Hamiltonian:
        """
        Creates a hamiltonian from a triangular lattice with SOC, exchange coupling and s-wave SC
        """
        if self.random_lattice:
            g = self.np_random.choice([geometry.square_lattice(),
                                       geometry.triangular_lattice(), geometry.honeycomb_lattice()])
        else:
            g = geometry.triangular_lattice()
        h = g.get_hamiltonian()
        h.add_rashba(1 * self.__random_factor())
        self.zeeman = z = 1 * self.__random_factor()
        h.add_zeeman([0, 0, z])
        h.add_swave(0.3 * self.__random_factor())
        return h

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Plots the Density of States:
        Returns:
            RGBA Array (1440, 2560, 4)
        """
        if self.render_mode is None:
            return None

        fig, ax = plt.subplots(figsize=(25.60, 14.40), dpi=100)
        (e, d) = self.get_obs(return_energy=True)
        plt.plot(e, d)
        plt.xlabel("Energy")
        plt.ylabel("DOS")
        plt.title("Density of states")

        fig.canvas.draw()
        l, b, w, h = fig.canvas.figure.bbox.bounds
        w, h = int(w), int(h)

        with BytesIO() as buf:
            fig.savefig(buf, format="rgba")
            buf.seek(0)
            data = np.frombuffer(buf.read(), dtype=np.uint8)
            data = data.reshape((h, w, -1))
        return data


if __name__ == '__main__':
    env = ChemPotentialEnv(time_penalty=0.0001, random_range=0.1, continuous=True)
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.sample().shape)
    check_env(env, warn=True)
    print("Done")
