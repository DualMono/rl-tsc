import numpy as np
from train import *
from sbx import PPO
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.env_constructors import make_zeeman, make_1d, make_conductance
import os

time_limit = 100

# env = make_zeeman(noise=0.05)()
# model = PPO.load(f"saves/zeeman/zeeman_jax_005noise.zip")

# env = make_1d(noise=0.05)()
# model = PPO.load(f"saves/moire/moire_jax_010noise.zip")

env_kwargs = dict(limit_actions=True)
env = make_conductance(noise=0.05, **env_kwargs)()
model = PPO.load(f"saves/conductance/conductance_jax_005noise.zip")


def get_final_value(env):
    total_reward = 0
    obs, _ = env.reset()
    for _ in range(time_limit):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, env.unwrapped.get_value()


if __name__ == "__main__":
    print("Starting evaluations")
    n_runs = 50
    results = []
    for _ in tqdm(range(n_runs)):
        results.append(get_final_value(env))

    rewards = np.array(results)

    print(f"Max value: {np.max(rewards[:, 1])}")
    print(f"Mean value: {np.mean(rewards[:, 1])}")
    print(f"Success rate: {np.mean(rewards[:, 1] >= 0.25)*100:.2f} %")

    plt.subplot(1, 2, 1)
    plt.hist(rewards[:, 0])
    plt.title("Reward")

    plt.subplot(1, 2, 2)
    plt.hist(rewards[:, 1])
    plt.title("Value")
    plt.show()
