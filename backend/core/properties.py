from dataclasses import dataclass
import numpy as np

# ===================================================================
#  AquiferProperties (layered hydraulic parameters)
#  Required by matrix assembly, transmissivity functions, etc.
# ===================================================================

@dataclass
class AquiferProperties:
    """
    Layered hydraulic properties used for saturated groundwater flow.

    Kx, Ky, Kz : arrays of shape (nlay, nx, ny)
    thickness  : array of shape (nlay,)
    Sy, Ss     : optional (storage values)
    """

    Kx: np.ndarray
    Ky: np.ndarray
    Kz: np.ndarray
    thickness: np.ndarray
    Sy: np.ndarray = None
    Ss: np.ndarray = None
    porosity: np.ndarray = None
    confined: bool = False

    def __post_init__(self):
        self.nlay, self.nx, self.ny = self.Kx.shape

        if self.Ky.shape != (self.nlay, self.nx, self.ny):
            raise ValueError("Ky has incompatible shape")

        if self.Kz.shape != (self.nlay, self.nx, self.ny):
            raise ValueError("Kz has incompatible shape")

        if len(self.thickness) != self.nlay:
            raise ValueError("Thickness array length must equal number of layers")

    # ------------------------------------------
    # Darcy-based transmissivity
    # ------------------------------------------
    def transmissivity_x(self, k, i, j):
        return self.Kx[k, i, j] * self.thickness[k]

    def transmissivity_y(self, k, i, j):
        return self.Ky[k, i, j] * self.thickness[k]
