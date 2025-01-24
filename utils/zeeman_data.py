from enum import Enum
import numpy as np
import os
from utils.path_utils import utils_dir


class Location(Enum):
    LEFT = 0
    MID = 1
    RIGHT = 2


class ZeemanChernData:
    def __init__(self):
        path = utils_dir

        self.chernGapLeft = np.load(f"{path}/chernGapLeft.npy")
        self.chernGapRight = np.load(f"{path}/chernGapRight.npy")
        self.chernGap = np.load(f"{path}/chernGap.npy")

        self.muMinL = -3.5
        self.muMaxL = 1
        self.muMinR = 5.5
        self.muMaxR = 6.5
        self.zeemanMin = 0
        self.zeemanMax = 0.4

        self.muMin = -3.5
        self.muMax = 6.5
        self.zeemanMin = 0.0
        self.zeemanMax = 0.3

    def __getitem__(self, item) -> tuple[float, float]:
        # return self.chernGap[self.get_index(*item)]
        mu, zeeman = item
        location, i, j = self.get_location_index(mu, zeeman)
        match location:
            case Location.LEFT:
                return self.chernGapLeft[j, i]
            case Location.MID:
                return 0, 0
            case Location.RIGHT:
                return self.chernGapRight[j, i]

    def get_location_index(self, mu, zeeman) -> tuple[Location, int, int]:
        """
        Converts a tuple of mu, zeeman values to location and index in the chernGap arrays
        """
        if mu < self.muMaxL:
            n = self.chernGapLeft.shape[0]
            i = np.clip(int((mu - self.muMinL) / (self.muMaxL - self.muMinL) * n), 0, n - 1)
            j = np.clip(int(zeeman / self.zeemanMax * n), 0, n - 1)
            return Location.LEFT, i, j
        elif mu <= self.muMinR:
            return Location.MID, 0, 0
        else:
            n = self.chernGapRight.shape[0]
            i = np.clip(int((mu - self.muMinR) / (self.muMaxR - self.muMinR) * n), 0, n - 1)
            j = np.clip(int(zeeman / self.zeemanMax * n), 0, n - 1)
            return Location.RIGHT, i, j

    def get_index(self, mu, zeeman) -> tuple[int, int]:
        shape = self.chernGap.shape
        i = np.clip(int((mu - self.muMin) / (self.muMax - self.muMin) * shape[0]), 0, shape[0] - 1)
        j = np.clip(int(zeeman / self.zeemanMax * shape[1]), 0, shape[1] - 1)
        return i, j


class ZeemanDosData:
    def __init__(self):
        path = utils_dir
        self.dos_array = np.load(f"{path}/dos.npy")
        self.muMin = -3.5
        self.muMax = 6.5
        self.zeemanMin = 0.0
        self.zeemanMax = 0.3

    def __getitem__(self, item):
        mu, zeeman = item
        i, j = self.get_index(mu, zeeman)
        return self.dos_array[i, j, :]

    def get_index(self, mu, zeeman) -> tuple[int, int]:
        shape = self.dos_array.shape
        i = np.clip(int((mu - self.muMin) / (self.muMax - self.muMin) * shape[0]), 0, shape[0] - 1)
        j = np.clip(int(zeeman / self.zeemanMax * shape[1]), 0, shape[1] - 1)
        return i, j


if __name__ == "__main__":
    chernGap = ZeemanChernData()
    import matplotlib.pyplot as plt
    n = 100
    mu = np.linspace(-4, 7, n)
    zeeman = np.linspace(0, 0.3, n)
    grid = np.meshgrid(mu, zeeman)
    grid = np.stack(grid, axis=-1)
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = chernGap[grid[i, j]][0]

    c = np.abs(np.round(c))
    plt.contourf(grid[:, :, 0], grid[:, :, 1], c)
    plt.show(block=False)

    from pyqula import geometry
    dos_data = ZeemanDosData()
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()

    mu = 3.5
    z = 0.01
    h.add_onsite(-mu)
    h.add_rashba(0.3)
    h.add_swave(0.1)
    h.add_zeeman([0, 0, z])

    print(f"Chern (Cached): {chernGap[mu, z][0]:.2f}")
    print(f"Chern (Calc): {h.get_chern():.2f}")

    e, d = h.get_dos(delta=5e-2, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=100)
    fig = plt.figure()
    plt.plot(e, d)
    plt.plot(e, dos_data[mu, z] + 0.5)
    plt.show()
