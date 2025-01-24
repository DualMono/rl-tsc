from __future__ import annotations
from typing import TYPE_CHECKING

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO

from utils.env_constructors import make_zeeman, make_1d, make_conductance
from utils.rl import DropoutPolicy

if TYPE_CHECKING:
    from envs.one_dim_env import OneDimEnv
    from envs.zeeman_potential_env import ZeemanPotentialEnv
    from envs.conductance_env import ConductanceEnv


class GUIEnvWrapper:
    def __init__(self, env: OneDimEnv | ZeemanPotentialEnv, model: PPO):
        self.env = env
        self.model = model
        self.obs = self.env.reset()[0]
        self.e = self.env.get_dos()[0]

    def reset(self) -> plt.Figure:
        """
        Reset the environment and returns the rendered image
        """
        self.obs = self.env.reset()[0]
        return self.render()

    def render(self) -> plt.Figure:
        fig = plt.figure()
        d = self.env.get_obs()
        plt.plot(self.e, d)
        plt.xlabel("Energy")
        plt.ylabel("DOS")
        plt.title(f"Topological invariant: {self.env.get_invariant():.2f} Gap: {self.env.get_gap():.2f}")
        fig.tight_layout()
        return fig

    def step_automatic(self):
        """
        Updates the environment using the trained model and returns the rendered image
        Overridden to ignore the stop signal
        """
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return self.render()

    def solve(self):
        """
        Solve the environment and returns the rendered image
        """
        for _ in range(10):
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        return self.render()

    def set_noise(self, noise: float):
        self.env.noise = noise
        return self.render()


class MoireEnvWrapper(GUIEnvWrapper):
    def __init__(self):
        env: OneDimEnv = make_1d(render_mode=None, n=2)().unwrapped
        model = PPO.load(f"saves/moire/moire_jax_010noise.zip", device="cpu",
                         custom_objects={"DropoutPolicy": DropoutPolicy})
        super().__init__(env, model)

    def step(self, potential, zeeman, modulation):
        action = np.array([
            potential,
            zeeman,
            modulation,
        ], dtype=np.float64)
        values = np.array([self.env.potential,
                           self.env.zeeman,
                           self.env.modulation], dtype=np.float64)
        action -= values
        action = self.env.unscale_action(action)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return self.obs, reward, terminated, truncated, info

    def update(self, potential: float, zeeman: float, modulation: float):
        """
        Update the environment with the given parameters and returns the rendered image
        """
        self.step(potential, zeeman, modulation)
        return self.render()


class ConductanceEnvWrapper:
    def __init__(self):
        self.env: ConductanceEnv = make_conductance(limit_actions=True)().unwrapped
        self.model = PPO.load(f"saves/conductance/conductance_jax_005noise.zip", device="cpu",
                              custom_objects={"DropoutPolicy": DropoutPolicy})
        self.obs = self.env.reset()[0]

    def reset(self) -> plt.Figure:
        """
        Reset the environment and returns the rendered image
        """
        self.obs = self.env.reset()[0]
        return self.render()

    def render(self) -> plt.Figure:
        fig = plt.figure()
        obs = self.env.get_obs()
        plt.plot(obs)
        plt.xlabel("Energy")
        plt.ylabel("Conductance")
        plt.title(f"Topological invariant: {self.env.get_invariant():.2f} Gap: {self.env.get_gap():.2f}")
        fig.tight_layout()
        return fig

    def step_automatic(self):
        """
        Updates the environment using the trained model and returns the rendered image
        Overridden to ignore the stop signal
        """
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return self.render()

    def solve(self):
        """
        Solve the environment and returns the rendered image
        """
        for _ in range(10):
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        return self.render()

    def set_noise(self, noise: float):
        self.env.noise = noise
        return self.render()

    def step(self, potential, zeeman, modulation):
        action = np.array([
            potential,
            zeeman,
            modulation,
        ], dtype=np.float64)
        values = np.array([self.env.potential,
                           self.env.zeeman,
                           self.env.modulation], dtype=np.float64)
        action -= values
        self.env.set_params(action)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return self.obs, reward, terminated, truncated, info

    def update(self, potential: float, zeeman: float, modulation: float):
        """
        Update the environment with the given parameters and returns the rendered image
        """
        self.step(potential, zeeman, modulation)
        return self.render()


class ZeemanEnvWrapper(GUIEnvWrapper):
    def __init__(self):
        env: ZeemanPotentialEnv = make_zeeman(render_mode=None)().unwrapped
        model = PPO.load(f"saves/zeeman/zeeman_jax_010noise.zip", device="cpu",
                         custom_objects={"DropoutPolicy": DropoutPolicy})
        super().__init__(env, model)

    def step(self, potential, zeeman):
        action = np.array([
            potential,
            zeeman,
        ], dtype=np.float64)
        values = np.array([
            self.env.potential,
            self.env.zeeman], dtype=np.float64)
        action -= values
        action = self.env.unscale_action(action)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return self.obs, reward, terminated, truncated, info

    def update(self, potential: float, zeeman: float):
        """
        Update the environment with the given parameters and returns the rendered image
        """
        self.step(potential, zeeman)
        return self.render()


def round2(x: float) -> float:
    return round(x, ndigits=2)


with gr.Blocks() as demo:
    conductance_env = ConductanceEnvWrapper()
    moire_env = MoireEnvWrapper()
    zeeman_env = ZeemanEnvWrapper()

    gr.Markdown(
        """
        # Topological Superconductor Design
        
        The sliders control the following parameters:
        - `Zeeman field` controls the exchange coupling strength.
        - `Chemical potential` shifts the Fermi level.
        - `Modulation strength` adds a periodic shift to the chemical potential. Only for the `1D` environment.
        
        The buttons:
        - `Manual Update` Sets the parameters to the values in the sliders.
        - `Automatic Update` Ignores the sliders and uses the trained models to update the environment once.
        - `Solve` Let's the trained agents fully "solve" the system.
        - `Reset` Resets and randomizes the environment state.
        """
    )

    with gr.Tab("Conductance"):
        with gr.Row():
            with gr.Column():
                conductance_limits = conductance_env.env.slider_limits
                zeeman_slider_c = gr.Slider(minimum=conductance_limits.zeeman_min,
                                            maximum=conductance_limits.zeeman_max,
                                            step=0.01, value=round2(conductance_env.env.zeeman), label="Zeeman field")
                potential_slider_c = gr.Slider(minimum=conductance_limits.potential_min,
                                               maximum=conductance_limits.potential_max,
                                               step=0.01, value=round2(conductance_env.env.potential),
                                               label="Chemical potential")
                modulation_slider_c = gr.Slider(minimum=conductance_limits.modulation_min,
                                                maximum=conductance_limits.modulation_max,
                                                step=0.01, value=round2(conductance_env.env.modulation),
                                                label="Modulation strength")

                update_button_c = gr.Button(value="Manual Update")
                help_button_c = gr.Button(value="Automatic Update")
                solve_button_c = gr.Button(value="Solve", variant="primary")
                reset_button_c = gr.Button(value="Reset", variant="stop")

                noise_slider_c = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=round2(conductance_env.env.noise),
                                           label="Noise scale")

            # Plot on the right
            image_plot_c = gr.Plot(value=conductance_env.render(), label="Observation")

        update_button_c.click(fn=conductance_env.update,
                              inputs=[potential_slider_c, zeeman_slider_c, modulation_slider_c],
                              outputs=[image_plot_c])
        help_button_c.click(fn=lambda: (conductance_env.step_automatic(), round2(conductance_env.env.zeeman),
                                        round2(conductance_env.env.potential), round2(conductance_env.env.modulation)),
                            outputs=[image_plot_c, zeeman_slider_c, potential_slider_c, modulation_slider_c])
        solve_button_c.click(
            fn=lambda: (
                conductance_env.solve(), round2(conductance_env.env.zeeman), round2(conductance_env.env.potential),
                round2(conductance_env.env.modulation)),
            outputs=[image_plot_c, zeeman_slider_c, potential_slider_c, modulation_slider_c])
        reset_button_c.click(
            fn=lambda: (
                conductance_env.reset(), round2(conductance_env.env.zeeman), round2(conductance_env.env.potential),
                round2(conductance_env.env.modulation)),
            outputs=[image_plot_c, zeeman_slider_c, potential_slider_c, modulation_slider_c])
        noise_slider_c.release(fn=conductance_env.set_noise, inputs=[noise_slider_c], outputs=[image_plot_c])

    with gr.Tab("1D"):
        with gr.Row():
            with gr.Column():
                moire_limits = moire_env.env.slider_limits
                zeeman_slider = gr.Slider(minimum=moire_limits.zeeman_min, maximum=moire_limits.zeeman_max,
                                          step=0.01, value=round2(moire_env.env.zeeman), label="Zeeman field")
                potential_slider = gr.Slider(minimum=moire_limits.potential_min, maximum=moire_limits.potential_max,
                                             step=0.01, value=round2(moire_env.env.potential),
                                             label="Chemical potential")
                modulation_slider = gr.Slider(minimum=moire_limits.modulation_min, maximum=moire_limits.modulation_max,
                                              step=0.01, value=round2(moire_env.env.modulation),
                                              label="Modulation strength")

                update_button = gr.Button(value="Manual Update")
                help_button = gr.Button(value="Automatic Update")
                solve_button = gr.Button(value="Solve", variant="primary")
                reset_button = gr.Button(value="Reset", variant="stop")

                noise_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=round2(moire_env.env.noise),
                                         label="Noise scale")

            # Plot on the right
            image_plot = gr.Plot(value=moire_env.render(), label="Observation")

        update_button.click(fn=moire_env.update, inputs=[potential_slider, zeeman_slider, modulation_slider],
                            outputs=[image_plot])
        help_button.click(fn=lambda: (moire_env.step_automatic(), round2(moire_env.env.zeeman),
                                      round2(moire_env.env.potential), round2(moire_env.env.modulation)),
                          outputs=[image_plot, zeeman_slider, potential_slider, modulation_slider])
        solve_button.click(
            fn=lambda: (moire_env.solve(), round2(moire_env.env.zeeman), round2(moire_env.env.potential),
                        round2(moire_env.env.modulation)),
            outputs=[image_plot, zeeman_slider, potential_slider, modulation_slider])
        reset_button.click(
            fn=lambda: (moire_env.reset(), round2(moire_env.env.zeeman), round2(moire_env.env.potential),
                        round2(moire_env.env.modulation)),
            outputs=[image_plot, zeeman_slider, potential_slider, modulation_slider])
        noise_slider.release(fn=moire_env.set_noise, inputs=[noise_slider], outputs=[image_plot])

    with gr.Tab("2D"):
        with gr.Row():
            with gr.Column():
                zeeman_limits = zeeman_env.env.slider_limits
                zeeman_slider2 = gr.Slider(minimum=zeeman_limits.zeeman_min, maximum=zeeman_limits.zeeman_max,
                                           step=0.01,
                                           value=round2(zeeman_env.env.zeeman), label="Zeeman field")
                potential_slider2 = gr.Slider(minimum=zeeman_limits.potential_min, maximum=zeeman_limits.potential_max,
                                              step=0.01, value=round2(zeeman_env.env.potential),
                                              label="Chemical potential")

                update_button2 = gr.Button(value="Manual Update")
                auto_button2 = gr.Button(value="Automatic update")
                solve_button2 = gr.Button(value="Solve", variant="primary")
                reset_button2 = gr.Button(value="Reset", variant="stop")

                noise_slider2 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=round2(zeeman_env.env.noise),
                                          label="Noise scale")

            # Plot on the right
            zeeman_plot = gr.Plot(value=zeeman_env.render(), label="Observation")

        reset_button2.click(
            fn=lambda: (zeeman_env.reset(), round2(zeeman_env.env.zeeman), round2(zeeman_env.env.potential)),
            outputs=[zeeman_plot, zeeman_slider2, potential_slider2])
        solve_button2.click(
            fn=lambda: (zeeman_env.solve(), round2(zeeman_env.env.zeeman), round2(zeeman_env.env.potential)),
            outputs=[zeeman_plot, zeeman_slider2, potential_slider2])
        auto_button2.click(fn=lambda: (zeeman_env.step_automatic(), round2(zeeman_env.env.zeeman),
                                       round2(zeeman_env.env.potential)),
                           outputs=[zeeman_plot, zeeman_slider2, potential_slider2])
        update_button2.click(fn=zeeman_env.update, inputs=[potential_slider2, zeeman_slider2],
                             outputs=[zeeman_plot])
        noise_slider2.release(fn=zeeman_env.set_noise, inputs=[noise_slider2], outputs=[zeeman_plot])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
