from pyqula import geometry
import numpy as np
import multiprocessing as mp
import itertools

from pyqula.hamiltonians import Hamiltonian
from tqdm import tqdm

g = geometry.triangular_lattice()

n_mu = 2001
n_zeeman = 121
n_hopping = 9
potential = np.linspace(-3.5, 6.5, n_mu)
zeeman = np.linspace(0, 0.3, n_zeeman)
hopping = np.linspace(-0.3, 0.3, n_hopping)
energies = np.linspace(-3.5, 3.5, 100)

hamiltonians: list[Hamiltonian] = [g.get_hamiltonian(tij=[1, tj]) for tj in hopping]
[h.add_rashba(0.3) for h in hamiltonians]
[h.add_swave(0.1) for h in hamiltonians]


def calc_data(index):
    i, j, k = index
    mu = -potential[i]
    z = zeeman[j]
    h = hamiltonians[k].copy()
    h.add_onsite(mu)
    h.add_zeeman([0, 0, z])
    chern = np.array((
        h.get_chern(nk=50),
        h.get_gap()
    ), dtype=np.float32)
    _, dos = h.get_dos(delta=5e-2, energies=energies, write=False, nk=100)
    dos = dos.astype(np.float32)
    return index, chern, dos


calculate = True

if __name__ == "__main__":
    if calculate:
        print("Starting calculations")
        chernGap = np.zeros((n_mu, n_zeeman, n_hopping, 2), dtype=np.float32)
        dosArray = np.zeros((n_mu, n_zeeman, n_hopping, 100), dtype=np.float32)
        inputs = list(itertools.product(range(n_mu), range(n_zeeman), range(n_hopping)))
        with mp.Pool() as pool:
            for index, chern, dos in tqdm(pool.imap_unordered(calc_data, inputs), total=len(inputs)):
                chernGap[index, :] = chern
                dosArray[index, :] = dos
        np.save("chernGapHopping.npy", chernGap)
        np.save("dosHopping.npy", dosArray)
        print("Done!")
    else:
        pass
