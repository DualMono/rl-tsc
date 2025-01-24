from pyqula import geometry
import numpy as np
import multiprocessing as mp
import itertools
from tqdm import tqdm
import os

g = geometry.triangular_lattice()
h_base = g.get_hamiltonian()
h_base.add_rashba(0.3)
h_base.add_swave(0.1)

n_mu = 2000
n_zeeman = 60
potential = np.linspace(-3.5, 6.5, n_mu)
zeeman = np.linspace(0, 0.3, n_zeeman)


def calc_data(i, j):
    mu = -potential[i]
    z = zeeman[j]
    h = h_base.copy()
    h.add_onsite(mu)
    h.add_zeeman([0, 0, z])
    chern = np.array((
        h.get_chern(nk=50),
        h.get_gap()
    ), dtype=np.float32)
    _, dos = h.get_dos(delta=5e-2, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=100)
    dos = dos.astype(np.float32)
    return i, j, chern, dos


def calc_star(args):
    return calc_data(*args)


calculate = True

if __name__ == "__main__":
    if calculate:
        print("Starting calculations")
        chernGap = np.zeros((n_mu, n_zeeman, 2), dtype=np.float32)
        dosArray = np.zeros((n_mu, n_zeeman, 100), dtype=np.float32)
        inputs = list(itertools.product(range(n_mu), range(n_zeeman)))
        num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", default=12))
        print("Starting pool")
        with mp.Pool(num_cpus) as pool:
            print("Pool started")
            for i, j, chern, dos in tqdm(pool.imap_unordered(calc_star, inputs), total=len(inputs)):
                print("Result")
                chernGap[i, j, :] = chern
                dosArray[i, j, :] = dos
        np.save("chernGap.npy", chernGap)
        np.save("dos.npy", dosArray)
    else:
        import matplotlib.pyplot as plt
        chernGap = np.load("chernGap.npy")
        dosArray = np.load("dos.npy")

        c = chernGap[:, :, 1] * np.invert(np.isclose(chernGap[:, :, 0], 0, atol=0.1))
        plt.contourf(c.T, levels=100, extent=[-3.5, 6.5, 0, 0.3])
        plt.show()
