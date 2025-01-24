from pathlib import Path

import gymnasium as gym

from envs.chem_potential_env import ChemPotentialEnv
from envs.conductance_env import ConductanceEnv
from envs.one_dim_env import OneDimEnv
from envs.zeeman_potential_env import ZeemanPotentialEnv


def make_zeeman(render_mode=None, noise=0.0, time_limit=10):
    def _init():
        env = ZeemanPotentialEnv(time_penalty=0.0, render_mode=render_mode, noise=noise)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, time_limit)
        return env

    return _init


def make_chem(render_mode=None, time_limit=10):
    def _init():
        env = ChemPotentialEnv(time_penalty=0.0, random_range=0.2, render_mode=render_mode, random_lattice=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, time_limit)
        return env

    return _init


def make_1d(render_mode=None, n=2, noise=0.0, time_limit=10):
    def _init():
        env = OneDimEnv(time_penalty=0.0, render_mode=render_mode, n=n, noise=noise)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, time_limit)
        return env

    return _init


def make_conductance(noise=0.0, time_limit=10, **kwargs):
    def _init():
        env = ConductanceEnv(noise=noise, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, time_limit)
        return env
    return _init


def get_latest_checkpoint():
    checkpoints = list(Path("./checkpoints").glob("*.zip"))
    if len(checkpoints) == 0:
        raise FileNotFoundError("No checkpoints found")
    return str(max(checkpoints, key=lambda x: x.stat().st_mtime))
