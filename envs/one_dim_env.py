from io import BytesIO
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pyqula import geometry
from pyqula.hamiltonians import Hamiltonian
from pyqula.geometry import Geometry
from pyqula.topology import berry_phase

from stable_baselines3.common.env_checker import check_env

from dataclasses import dataclass

# mpl.use("tkagg")


@dataclass
class FigContent:
    """
    Container class for figure content
    """
    dos: plt.Line2D | None = None
    zeeman: Slider | None = None
    modulation: Slider | None = None
    potential: Slider | None = None
    ax: plt.Axes | None = None


@dataclass
class SliderLimits:
    """
    Container class for figure slider limits
    """
    zeeman_min = 0.0
    zeeman_max = 0.4
    potential_min = 0
    potential_max = 3
    modulation_min = -3
    modulation_max = 3


class OneDimEnv(gym.Env):
    """
    Observation space:
        Density of states (100,) array
    Actions (Continuous):
        0: Change the chemical potential
        1: Change the Zeeman field
        2: Change the modulation strength
    Reward:
        Change in state value. See OneDimEnv.get_value
    Params:
        time_penalty: The time penalty for each step
        render_mode: The render mode, either "human" or "rgb_array"
    """
    metadata = {'render_modes': ["rgb_array", "human"], "render_fps": 2}

    def __init__(self, random_range=0.0, time_penalty=0.0, obs_size=100, render_mode=None, n=2, noise=0.0):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Unsupported render mode"

        self.render_mode = render_mode
        self.fig: plt.Figure | None = None
        self.fig_content = FigContent()
        self.slider_limits = SliderLimits()
        self.dpi = 100
        self.dpi_scale = 4
        self.pause_time = 1.5

        self.obs_size = obs_size
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(obs_size,), dtype=np.float32)

        self.random_range = random_range
        self.time_penalty = time_penalty
        self.noise = noise

        self.h: Hamiltonian | None = None
        self.lattice = geometry.chain().get_supercell(n)
        self.n: int = n
        self.potential = 0.0
        self.zeeman = 0.0
        self.modulation = 0.0
        self.__old_value = 0.0
        self.__berry = 0.0
        self.__gap = 0.0
        self.__old_obs: np.ndarray | None = None

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

        return obs, {"chern": self.__berry, "gap": self.__gap}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.h is not None, "You forgot to call reset()"
        action = self.scale_action(action)

        if self.__check_limits(action):
            return self.__old_obs, -0.5, True, False, {"chern": self.__berry, "gap": self.__gap}

        self.h.add_onsite(action[0])
        self.potential += action[0]
        self.h.add_zeeman(action[1])
        self.zeeman += action[1]
        self.h.add_onsite(lambda r: action[2] * np.cos(2 * np.pi / self.n * (r[0] - self.lattice.r[0][0])))
        self.modulation += action[2]

        obs = self.get_obs()
        # Reward is the change in the state value
        new_value = self.get_value()
        reward = new_value - self.__old_value
        self.__old_value = new_value

        terminated = bool(new_value > 0.25)
        truncated = False
        info = {"chern": self.__berry, "gap": self.__gap}
        reward -= self.time_penalty

        if self.render_mode == "human":
            self.__update_figure()

        return obs, reward, terminated, truncated, info

    def __check_limits(self, action: ActType) -> bool:
        """
        Checks if applying the action stays within physical limits. Returns True if the action goes out of bounds.
        """
        potential_check = 0 <= self.potential + action[0] <= 1.5
        zeeman_check = 0 <= self.zeeman + action[1] <= 0.3
        return not all([potential_check, zeeman_check])

    def get_value(self) -> float:
        """
        Returns the value of the current state.
        The value is defined as |berry| * gap
        """
        value = 0
        self.__gap = gap = self.h.get_gap()
        if gap > 0:
            self.__berry = berry = np.abs(berry_phase(self.h))
            value += berry * gap
        # Penalize going over the limits. Bad results so terminate the episode instead
        potential_penalty = np.max([-10 * self.potential, 0, 10 * (self.potential - 1.5)])
        zeeman_penalty = np.max([-10 * self.zeeman, 0, 10 * (self.zeeman - 0.3)])
        return value

    def get_dos(self) -> tuple[np.ndarray, np.ndarray]:
        return self.h.get_dos(delta=5e-2, energies=np.linspace(-3.5, 3.5, self.obs_size), write=False, nk=100)

    def get_obs(self) -> np.ndarray:
        """
        Returns the observation
        """
        (e, dos) = self.get_dos()
        dos += self.np_random.normal(size=dos.shape, scale=self.noise)
        dos = np.maximum(dos, 0.0)
        dos: np.ndarray = dos.astype(np.float32)
        return dos

    @staticmethod
    def scale_action(action: ActType) -> ActType:
        """
        Scales the action
        """
        action[0] *= 1
        action[1] *= 0.2
        action[2] *= 1
        return action

    @staticmethod
    def unscale_action(action: ActType) -> ActType:
        """
        Unscales the action
        """
        action[0] *= 1
        action[1] *= 5
        action[2] *= 1
        return action

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

    def __init_hamiltonian(self) -> Hamiltonian:
        """
        Creates a hamiltonian from a triangular lattice with SOC, exchange coupling and s-wave SC
        """
        h = self.lattice.get_hamiltonian()  # get the Hamiltonian
        n = self.n
        g = self.lattice

        # add onsite modulation
        self.modulation = modulation = self.__random_between(0, 0.2)
        h.add_onsite(lambda r: modulation * np.cos(2 * np.pi / n * (r[0] - g.r[0][0])))
        self.potential = potential = self.__random_between(0, 1)
        h.add_onsite(potential)  # add chemical potential

        h.add_rashba(0.3)  # add Rashba SOC
        self.zeeman = zeeman = self.__random_between(0.0, 0.2)
        h.add_zeeman([0, 0, zeeman])  # add Zeeman field
        h.add_swave(0.1)  # add superconducting pairing

        return h

    def get_gap(self) -> float:
        return self.h.get_gap()

    def get_invariant(self) -> float:
        if self.get_gap() > 0:
            return np.abs(berry_phase(self.h))
        else:
            return 0.0

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
        gap = self.h.get_gap()
        if gap > 0:
            berry = np.abs(berry_phase(self.h))
        else:
            berry = 0
        ax.set_title(f"Gap: {gap:.2f} Berry: {berry / np.pi:.2f} π")

        self.fig_content.potential.set_val(self.potential)
        self.fig_content.zeeman.set_val(self.zeeman)
        self.fig_content.modulation.set_val(self.modulation)

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

        axModulation = fig.add_axes([right_margin + 0.3, 0.25, 0.0225, 0.5])
        modulation_slider = Slider(
            ax=axModulation,
            label='Modulation',
            valmin=self.slider_limits.modulation_min,
            valmax=self.slider_limits.modulation_max,
            valinit=self.modulation,
            orientation="vertical",
            valfmt='%1.2f'
        )
        axModulation.add_artist(axModulation.yaxis)
        axModulation.yaxis.set_visible(True)
        axModulation.yaxis.set_ticks(np.linspace(self.slider_limits.modulation_min,
                                                 self.slider_limits.modulation_max, 3))
        self.fig_content.modulation = modulation_slider

        gap = self.h.get_gap()
        if gap > 0:
            berry = np.abs(berry_phase(self.h))
        else:
            berry = 0
        ax.set_title(f"Gap: {gap:.2f} Berry: {berry / np.pi:.2f} π")
        fig.canvas.draw()
        return fig

    def close(self):
        if self.render_mode == "human":
            plt.pause(2)
        plt.close(self.fig)


if __name__ == '__main__':
    env = OneDimEnv()
    print("Env dimensions")
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.sample().shape)
    print("Checking environment")
    check_env(env, warn=True)
    print("Done")
