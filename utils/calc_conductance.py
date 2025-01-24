from dataclasses import dataclass

from pyqula import geometry
import numpy as np
import multiprocessing as mp
import itertools

from pyqula.hamiltonians import Hamiltonian
from pyqula.topology import berry_phase
from pyqula.transporttk.localprobe import LocalProbe
from tqdm import tqdm
import pickle
import os
import time


@dataclass
class State:
    potential: float
    zeeman: float
    modulation: float
    berry: float
    gap: float
    obs: np.ndarray


n = 2
energies = np.linspace(-1.4, 1.4, 100)

g = geometry.chain().get_supercell(n)
hamiltonian = g.get_hamiltonian()
hamiltonian.add_rashba(0.3)
hamiltonian.add_swave(0.1)

n_mu = 301
n_zeeman = 61
n_moire = 301
potential = np.linspace(0, 1.5, n_mu)
zeeman = np.linspace(0, 0.3, n_zeeman)
moire = np.linspace(0, 1.5, n_moire)

values = np.array(np.meshgrid(potential, zeeman, moire)).T.reshape(-1, 3)
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", default=0))
total_jobs = int(os.getenv("SLURM_ARRAY_TASK_COUNT", default=1))

split_values = np.array_split(values, total_jobs)
values = split_values[job_id]
total = values.shape[0]


def calc_state(index: int) -> tuple[int, State]:
    value = values[index]
    mu, z, modulation = value
    h = hamiltonian.copy()

    h.add_zeeman([0, 0, z])
    h.add_onsite(lambda r: modulation * np.cos(2 * np.pi / n * (r[0] - g.r[0][0])))
    h.add_onsite(mu)

    gap = h.get_gap()
    berry = np.abs(berry_phase(h)) if gap > 0 else 0.0

    lp = LocalProbe(h)
    didv_weak = np.array([lp.didv(energy=e, T=0.1) for e in energies])
    didv_strong = np.array([lp.didv(energy=e, T=1.0) for e in energies])
    obs = np.concatenate((didv_weak, didv_strong), axis=0).astype(np.float32)

    state = State(mu, z, modulation, berry, gap, obs)
    return index, state


calculate = False

if __name__ == "__main__":
    if calculate:
        print("Starting calculations")
        import sys
        print(f"{sys.executable=}")
        berry_gap = np.zeros((total, 2), dtype=np.float32)
        obs_array = np.zeros((total, 200), dtype=np.float32)
        inputs = range(total)
        berry_name = f"out/conductanceBerry{job_id}.npy"
        obs_name = f"out/conductanceObs{job_id}.npy"
        start_time = time.time()

        num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", default=mp.cpu_count()))
        print(f"Starting pool with {num_cpus} CPUs")
        with mp.Pool(num_cpus) as pool:
            print("Pool started")
            for index, state in tqdm(pool.imap_unordered(calc_state, inputs), total=len(inputs)):
                berry_gap[index, :] = state.berry_array, state.gap
                obs_array[index, :] = state.obs
                if index % 100 == 0:
                    np.save(berry_name, berry_gap)
                    np.save(obs_name, obs_array)
        np.save(berry_name, berry_gap)
        np.save(obs_name, obs_array)
        print("Done!")
    else:
        # Test that all the data is there
        total_jobs = 10
        missing_indexes = {i: [] for i in range(total_jobs)}
        print("Testing data")
        for file in range(total_jobs):
            obs_name = f"out/conductanceObs{file}.npy"
            print(f"Testing file: {obs_name}")
            obs_array = np.load(obs_name)
            for index in range(obs_array.shape[0]):
                if np.all(obs_array[index, :] == 0):
                    missing_indexes[file].append(index)
        for file, indexes in missing_indexes.items():
            if indexes:
                print(f"File: {file} is missing {len(indexes)} entries")

        values = np.array(np.meshgrid(potential, zeeman, moire)).T.reshape(-1, 3)
        split_values = np.array_split(values, total_jobs)

        missing_job_ids = [job_id for job_id, indexes in missing_indexes.items() if indexes]
        for job_id in missing_job_ids:
            values = split_values[job_id]
            total = values.shape[0]

            obs_name = f"out/conductanceObs{job_id}.npy"
            berry_name = f"out/conductanceBerry{job_id}.npy"
            obs_array = np.load(obs_name)
            berry_gap = np.load(berry_name)
            inputs = missing_indexes[job_id]

            print(f"Recalculating job: {job_id}")
            with mp.Pool() as pool:
                for index, state in tqdm(pool.imap(calc_state, inputs), total=len(inputs)):
                    berry_gap[index, :] = state.berry_array, state.gap
                    obs_array[index, :] = state.obs
            np.save(berry_name, berry_gap)
            np.save(obs_name, obs_array)
        print("Done!")

