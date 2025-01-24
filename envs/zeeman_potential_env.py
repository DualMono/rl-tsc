from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, SupportsFloat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

from pyqula import geometry
from pyqula.geometry import Geometry
from pyqula.hamiltonians import Hamiltonian

from stable_baselines3.common.env_checker import check_env

from utils.zeeman_data import ZeemanChernData, ZeemanDosData


@dataclass
class FigContent:
    """
    Container class for figure content
    """
    dos: plt.Line2D | None = None
    zeeman: Slider | None = None
    potential: Slider | None = None
    ax: plt.Axes | None = None


@dataclass
class SliderLimits:
    """
    Container class for figure slider limits
    """
    zeeman_min = 0.0
    zeeman_max = 0.3
    potential_min = -3.5
    potential_max = 6.5


class LatticeTypes(Enum):
    """
    Enum for the lattice types
    """
    TRIANGULAR = 0
    SQUARE = 1
    HONEYCOMB = 2
    CACHED_TRIANGULAR = 3


class ZeemanPotentialEnv(gym.Env):
    """
    Observation space:
        Density of states: (100,) array
    Actions (Continuous):
        0: Change the chemical potential
        1: Change the Zeeman field
    Reward:
        Change in |C| + gap * |C|

    Params:
        time_penalty: Applies an optional penalty each timestep
        render_mode: "human" or "rgb_array"
        lattice: The lattice to use
        noise: How much noise to add to the observation
        obs_size: Controls the size of the observation. Default 100
    """
    metadata = {'render_modes': ["rgb_array", "human"], "render_fps": 2}

    def __init__(self, time_penalty=0.0, render_mode=None, lattice=LatticeTypes.CACHED_TRIANGULAR, noise=0.0,
                 obs_size=100):
        super().__init__()
        self.render_mode = render_mode
        self.fig: plt.Figure | None = None
        self.fig_content = FigContent()
        self.slider_limits = SliderLimits()
        self.dpi = 100
        self.dpi_scale = 4
        self.pause_time = 1.5

        self.obs_size = obs_size
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        self.time_penalty = time_penalty
        self.noise = noise

        self.chernData = ZeemanChernData()
        self.dosData = ZeemanDosData()
        self.h: Hamiltonian | None = None
        self.lattice: Geometry | None = None
        self.lattice_type = lattice
        self.potential = 0.0
        self.zeeman = 0.0
        self.__old_value = 0.0
        self.__old_obs: np.ndarray | None = None
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

        obs = self.get_obs()
        self.__old_obs = obs
        self.__old_value = self.get_value()

        if self.fig is None:
            if self.render_mode == "human":
                self.fig = self.__init_figure()
                self.fig.canvas.draw()
                self.fig.show()
                plt.pause(self.pause_time)
            if self.render_mode == "rgb_array":
                self.fig = self.__init_figure()
        else:
            self.__update_figure()

        return obs, {"chern": self.__chern, "gap": self.__gap}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.h is not None or self.lattice_type == LatticeTypes.CACHED_TRIANGULAR, "You forgot to call reset()"
        action = self.scale_action(action)

        if self.__check_limits(action):
            return self.__old_obs, -0.5, True, False, {"chern": self.__chern, "gap": self.__gap}

        self.potential += action[0]
        self.zeeman += action[1]
        if self.lattice_type != LatticeTypes.CACHED_TRIANGULAR:
            self.h.add_zeeman(action[1])
            self.h.add_onsite(-action[0])

        obs = self.get_obs()
        # Reward is the change in the state value
        new_value = self.get_value()
        reward = new_value - self.__old_value
        self.__old_value = new_value
        terminated = bool(new_value > 1)
        truncated = False
        info = {"chern": self.__chern, "gap": self.__gap}
        # Penalize taking long if early stopping is an option
        reward -= self.time_penalty

        if self.render_mode == "human":
            self.__update_figure()

        return obs, reward, terminated, truncated, info

    def get_value(self) -> float:
        """
        Returns the value of the current state. The reward is defined as |C| + gap * |C|
        """
        self.__chern = c = self.get_invariant()
        self.__gap = g = self.get_gap()
        return c + g * c

    def get_dos(self) -> tuple[np.ndarray, np.ndarray]:
        e = np.linspace(-3.5, 3.5, self.obs_size)
        if self.lattice_type == LatticeTypes.CACHED_TRIANGULAR:
            return e, self.dosData[self.potential, self.zeeman]
        else:
            return self.h.get_dos(delta=5e-2, energies=e, write=False, nk=100)

    def get_obs(self) -> np.ndarray:
        """
        Returns the observation
        """
        (e, dos) = self.get_dos()
        dos: np.ndarray = dos.astype(np.float32)
        dos *= (1 + self.np_random.normal(size=dos.shape, scale=self.noise) )  # renormalized noie
        dos = dos / (np.max(dos) + 1e-6)  # normalize obs
        return dos

    def __init_hamiltonian(self) -> Hamiltonian | None:
        """
        Creates a hamiltonian from a triangular lattice with SOC, exchange coupling and s-wave SC
        """
        self.potential = potential = self.__random_between(-2, 5)
        self.zeeman = zeeman = self.__random_between(0.0, 0.2)

        if self.lattice_type == LatticeTypes.CACHED_TRIANGULAR:
            return None
        if self.lattice is None:
            if self.lattice_type == LatticeTypes.TRIANGULAR:
                self.lattice = geometry.triangular_lattice()
            elif self.lattice_type == LatticeTypes.SQUARE:
                self.lattice = geometry.square_lattice()
            elif self.lattice_type == LatticeTypes.HONEYCOMB:
                self.lattice = geometry.honeycomb_lattice()

        h = self.lattice.get_hamiltonian()  # get the Hamiltonian

        h.add_rashba(0.3)  # add Rashba SOC
        h.add_swave(0.1)  # add superconducting pairing

        h.add_onsite(-potential)  # add chemical potential
        h.add_zeeman([0, 0, zeeman])  # add Zeeman field

        return h

    def get_gap(self) -> float:
        """
        Calculates the topological gap
        """
        if self.lattice_type == LatticeTypes.CACHED_TRIANGULAR:
            return self.chernData[self.potential, self.zeeman][1]
        else:
            return self.h.get_gap()

    def get_invariant(self) -> float:
        """
        Returns |C| rounded to the nearest int
        """
        if self.lattice_type == LatticeTypes.CACHED_TRIANGULAR:
            return np.abs(np.round(self.chernData[self.potential, self.zeeman], 0))[0]
        else:
            return np.abs(np.round(self.h.get_chern(nk=50), 0))

    def __random_between(self, low: float, high: float) -> float:
        """
        Returns a random number from the uniform distribution [low, high]
        """
        return self.np_random.uniform(low, high)

    @staticmethod
    def scale_action(action: ActType) -> ActType:
        """
        Scales the action
        """
        action[0] *= 2
        action[1] *= 0.1
        return action

    @staticmethod
    def unscale_action(action: ActType) -> ActType:
        """
        Unscales the action
        """
        action[0] *= 0.5
        action[1] *= 10
        return action

    def __check_limits(self, action: ActType) -> bool:
        """
        Returns true if the action would go over a limit
        """
        zeeman_limit = 0 <= self.zeeman + action[1] <= 0.3
        return not all([zeeman_limit])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Plots the Density of States
        Returns:
            RGBA Array (H, W, 4)
        """
        if self.render_mode == "rgb_array":
            self.__update_figure()
            fig = self.fig
            l, b, w, h = fig.canvas.figure.bbox.bounds
            w, h = int(w * self.dpi_scale), int(h * self.dpi_scale)
            with BytesIO() as buf:
                fig.savefig(buf, format="rgba", dpi=self.dpi * self.dpi_scale)
                buf.seek(0)
                data = np.frombuffer(buf.read(), dtype=np.uint8)
                data = data.reshape((h, w, 4))
            return data

    def __update_figure(self) -> None:
        """
        Updates the figure with the current state
        """
        (e, d) = self.get_dos()
        self.fig_content.dos.set_data(e, d)
        self.fig_content.ax.set_ylim(top=np.max(d) * 1.1)

        ax = self.fig.axes[0]
        chern = np.abs(self.get_invariant())
        gap = self.get_gap()
        ax.set_title(f"Gap: {gap:.2f} Chern: {chern:.2f}")

        self.fig_content.potential.set_val(self.potential)
        self.fig_content.zeeman.set_val(self.zeeman)

        self.fig.canvas.draw()
        if self.render_mode == "human":
            plt.pause(self.pause_time)

    def __init_figure(self) -> plt.Figure:
        """
        Plots the Density of States
        Returns:
            MPL Figure
        """
        assert self.fig is None, "Figure already exists"
        fig, ax = plt.subplots(figsize=(12, 5), dpi=self.dpi)
        fig: plt.Figure
        ax: plt.Axes
        (e, d) = self.get_dos()
        dos, = ax.plot(e, d)
        ax.set_xlabel("Energy")
        ax.set_ylabel("DOS")
        self.fig_content.dos = dos
        self.fig_content.ax = ax

        # adjust the main plot to make room for the sliders
        right_margin = 0.6
        fig.subplots_adjust(left=0.1, right=right_margin, bottom=0.1)

        # Make vertical sliders on the right side of the plot
        axZeeman = fig.add_axes([right_margin + 0.1, 0.25, 0.0225, 0.5])
        zeeman_slider = Slider(
            ax=axZeeman,
            label='Zeeman',
            valmin=self.slider_limits.zeeman_min,
            valmax=self.slider_limits.zeeman_max,
            valinit=self.zeeman,
            orientation="vertical",
            valfmt='%1.2f'
        )
        axZeeman.add_artist(axZeeman.yaxis)
        axZeeman.yaxis.set_visible(True)
        axZeeman.yaxis.set_ticks(np.linspace(self.slider_limits.zeeman_min, self.slider_limits.zeeman_max, 3))
        self.fig_content.zeeman = zeeman_slider

        axPotential = fig.add_axes([right_margin + 0.2, 0.25, 0.0225, 0.5])
        potential_slider = Slider(
            ax=axPotential,
            label='Potential',
            valmin=self.slider_limits.potential_min,
            valmax=self.slider_limits.potential_max,
            valinit=self.potential,
            orientation="vertical",
            valfmt='%1.2f'
        )
        axPotential.add_artist(axPotential.yaxis)
        axPotential.yaxis.set_visible(True)
        axPotential.yaxis.set_ticks(np.linspace(self.slider_limits.potential_min, self.slider_limits.potential_max, 3))
        self.fig_content.potential = potential_slider

        chern = np.abs(self.get_invariant())
        gap = self.get_gap()
        ax.set_title(f"Gap: {gap:.2f} Chern: {chern:.2f}")
        fig.canvas.draw()
        return fig

    def close(self):
        if self.render_mode == "human":
            plt.pause(2)
        plt.close(self.fig)


if __name__ == '__main__':
    env = ZeemanPotentialEnv(noise=0.1, lattice=LatticeTypes.CACHED_TRIANGULAR)
    print("Env dimensions")
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.sample().shape)
    print("Checking environment")
    check_env(env, warn=True)
    print("Done")
