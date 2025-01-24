import os
import multiprocessing
from dataclasses import dataclass, field
from typing import Callable

from sbx import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.env_constructors import *
from utils.rl import ParamCallbackPPO, DropoutPolicy


@dataclass
class Parameters:
    """
    Container class for parameters
    """
    num_envs: int
    time_steps: int

    n_steps: int
    batch_size: int
    ent_coef: float
    vf_coef: float
    lr: float
    policy_kwargs: dict

    env_constructor: Callable
    folder_name: str
    log_name: str
    env_kwargs: dict = field(default_factory=dict)


conductance_params = Parameters(
    num_envs=12,
    time_steps=int(1e6),
    n_steps=128 * 2,
    batch_size=32 * 2,
    ent_coef=0.01,
    vf_coef=0.5,
    lr=3e-4,
    policy_kwargs=dict(net_arch=[128] * 5),
    env_constructor=make_conductance,
    folder_name="conductance",
    log_name="conductance",
    env_kwargs=dict(limit_actions=True),
)

zeeman_params = Parameters(
    num_envs=int(os.getenv("SLURM_CPUS_PER_TASK", default=12)),
    time_steps=int(2e5),
    n_steps=128,
    batch_size=32,
    ent_coef=0.01,
    vf_coef=0.5,
    lr=3e-4,
    policy_kwargs=dict(net_arch=[128] * 3),
    env_constructor=make_zeeman,
    folder_name="zeeman",
    log_name="zeeman",
)

moire_params = Parameters(
    num_envs=int(os.getenv("SLURM_CPUS_PER_TASK", default=12)),
    time_steps=int(5e5),
    n_steps=128,
    batch_size=32,
    ent_coef=0.01,
    vf_coef=0.5,
    lr=3e-4,
    policy_kwargs=dict(net_arch=[128] * 4),
    env_constructor=make_1d,
    folder_name="moire",
    log_name="moire",
)


def eval_model(model: PPO, vec_env) -> float:
    """
    Returns the mean reward of the model in evaluation mode
    """
    from stable_baselines3.common.evaluation import evaluate_policy
    mean, std = evaluate_policy(model, vec_env, n_eval_episodes=1000)
    return mean


def save_model(model: PPO, path: str, vec_env) -> None:
    """
    Saves the model to disk. If a model already exists, saves the better one.
    """
    if os.path.exists(path):
        other_model = PPO.load(path, ustom_objects={"DropoutPolicy": DropoutPolicy})
        other_reward = eval_model(other_model, vec_env)
        this_reward = eval_model(model, vec_env)
        if this_reward > other_reward:
            model.save(path)
    else:
        model.save(path)


if __name__ == "__main__":
    params = zeeman_params
    training_runs = 10

    num_envs = params.num_envs
    time_steps = params.time_steps

    n_steps = params.n_steps
    batch_size = params.batch_size
    ent_coef = params.ent_coef
    vf_coef = params.vf_coef
    lr = params.lr
    policy_kwargs = params.policy_kwargs
    env_constructor = params.env_constructor
    folder_name = params.folder_name
    file_name = params.log_name + "_jax"
    env_kwargs = params.env_kwargs

    noise_levels = [0.05, 0.1, 0.2, 0.3]
    # noise_levels = [0.1]

    # Save a checkpoint every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1),
        save_path="./checkpoints",
    )
    # Save the hyperparameters every when training starts
    param_callback = ParamCallbackPPO()

    print("Creating envs")

    vec_env = DummyVecEnv([env_constructor(noise=0.0, **env_kwargs) for _ in range(num_envs)])

    print("Starting training")
    for noise in noise_levels:
        log_name = f"{file_name}_{noise * 100:0>3.0f}noise"
        save_path = f"saves/{folder_name}/{log_name}"
        vec_env.set_attr("noise", noise)

        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"./logs/{folder_name}",
                    n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef, vf_coef=vf_coef,
                    learning_rate=lr, policy_kwargs=policy_kwargs, device="cpu")

        model.learn(total_timesteps=time_steps, log_interval=10,
                    tb_log_name=log_name, callback=[checkpoint_callback, param_callback])

        print(f"Evaluating and saving model")
        save_model(model, save_path, vec_env)
        # Remove checkpoints after saving
        for file in os.listdir("./checkpoints"):
            os.remove(os.path.join("./checkpoints", file))

        # Repeat training for more logs
        for _ in range(training_runs - 1):
            model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"./logs/{folder_name}",
                        n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef, vf_coef=vf_coef,
                        learning_rate=lr)

            model.learn(total_timesteps=time_steps, log_interval=10,
                        tb_log_name=log_name)

    print("Done!")
