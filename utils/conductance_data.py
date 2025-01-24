"""
Module for loading and accessing the conductance data
"""

import numpy as np
from utils.path_utils import out_dir

out_dir = str(out_dir)  # Required for numba

n_potential = 301
n_zeeman = 61
n_moire = 301
potential_line = np.linspace(0, 1.5, n_potential, dtype=np.float32)
zeeman_line = np.linspace(0, 0.3, n_zeeman, dtype=np.float32)
moire_line = np.linspace(0, 1.5, n_moire, dtype=np.float32)


class __ConductanceData:
    def __init__(self):
        self.conductance = None
        self.berry_array = None

    def load_data(self):
        if self.conductance is None:
            self.conductance = np.concatenate(
                [
                    np.load(f"{out_dir}/conductanceObs{i}.npy", allow_pickle=True)
                    for i in range(10)
                ],
                axis=0,
            ).reshape((n_moire, n_potential, n_zeeman, 200))
            self.berry_array = np.concatenate(
                [
                    np.load(f"{out_dir}/conductanceBerry{i}.npy", allow_pickle=True)
                    for i in range(10)
                ],
                axis=0,
            ).reshape((n_moire, n_potential, n_zeeman, 2))

    def assert_data_is_loaded(self) -> None:
        if self.conductance is None:
            raise RuntimeError("Need to call load_data")

    def get_observation(self, mu: float, zeeman: float, moire: float) -> np.ndarray:
        """
        Returns the obs corresponding to the given mu, zeeman, moire values
        """
        self.assert_data_is_loaded()
        # Moire is mirrored for negative values
        moire = np.abs(moire)

        mu_index = self.float_to_index(mu, 0.0, 1.5, n_potential)
        zeeman_index = self.float_to_index(zeeman, 0.0, 0.3, n_zeeman)
        moire_index = self.float_to_index(moire, 0.0, 1.5, n_moire)

        return self.conductance[moire_index, mu_index, zeeman_index, :]

    def get_berry(self, mu: float, zeeman: float, moire: float) -> tuple[float, float]:
        """
        Returns the berry and gap corresponding to the given mu, zeeman, moire values
        """
        self.assert_data_is_loaded()
        # Moire is mirrored for negative values
        moire = np.abs(moire)

        mu_index = self.float_to_index(mu, 0.0, 1.5, n_potential)
        zeeman_index = self.float_to_index(zeeman, 0.0, 0.3, n_zeeman)
        moire_index = self.float_to_index(moire, 0.0, 1.5, n_moire)

        berry = self.berry_array[moire_index, mu_index, zeeman_index, 0]
        gap = self.berry_array[moire_index, mu_index, zeeman_index, 1]
        return berry, gap

    def float_to_index(self, x: float, minimum: float, maximum: float, n: int) -> int:
        """
        Converts a float to an index
        """
        return self.clip(int((x - minimum) / (maximum - minimum) * n), 0, n - 1)

    def clip(self, x: int, minimum: int, maximum: int) -> int:
        """
        Clips an integer to the given range
        """
        return min(max(x, minimum), maximum)


conductanceData = __ConductanceData()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import process_time
    from pyqula import geometry
    from pyqula.transporttk.localprobe import LocalProbe

    data = conductanceData
    data.load_data()
    _ = data.get_observation(0, 0, 0)
    start = process_time()
    data_amount = int(1e6)
    for i in range(data_amount):
        _ = data.get_observation(*np.random.rand(3))
    print(f"Data retrieved at: {data_amount / (process_time() - start):.3e} fps")
    exit(0)  # Exit here to only benchmark the data retrieval and skip the plots

    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian()  # get the Hamiltonian
    n = 2

    # Hamiltonian parameters
    potential = 0.5
    zeeman = 0.15
    modulation = 0.6

    # Create the Hamiltonian
    h.add_onsite(lambda r: modulation * np.cos(2 * np.pi / n * (r[0] - g.r[0][0])))
    h.add_onsite(potential)
    h.add_rashba(0.3)  # add Rashba SOC
    h.add_zeeman([0, 0, zeeman])  # add Zeeman field
    h.add_swave(0.1)  # add superconducting pairing

    # Calculations
    e = np.linspace(-1.4, 1.4, 100)
    lp = LocalProbe(h)
    obs = np.array([lp.didv(energy=e, T=0.1) for e in e])
    obs = np.concatenate((obs, np.array([lp.didv(energy=e, T=1.0) for e in e])), axis=0)
    obs /= np.max(obs)

    # Get the data from the saved files
    data_obs = data.get_observation(potential, zeeman, modulation)
    data_obs /= np.max(data_obs)

    berry_cached, gap_cached = data.get_berry(potential, zeeman, modulation)
    gap = h.get_gap()

    fig = plt.figure()
    plt.plot(obs, label="env")
    plt.plot(data_obs + 0.05, label="data")
    plt.legend()
    plt.title(f"Gap (Calc): {gap:.2f}, Gap (Cached): {gap_cached:.2f}")
    plt.show()
