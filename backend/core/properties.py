from dataclasses import dataclass, field
import numpy as np

@dataclass
class AquiferProperties:
    Kx: np.ndarray      # shape (nlay, nx, ny)
    Ky: np.ndarray
    Kz: np.ndarray
    thickness: np.ndarray    # shape (nlay,)
    Sy: np.ndarray = None
    Ss: np.ndarray = None
    porosity: np.ndarray = None
    confined: bool = False

    def __post_init__(self):
        # Validate dimensions
        self.nlay, self.nx, self.ny = self.Kx.shape

        if self.Ky.shape != (self.nlay, self.nx, self.ny):
            raise ValueError("Ky has incompatible shape")
        if self.Kz.shape != (self.nlay, self.nx, self.ny):
            raise ValueError("Kz has incompatible shape")

        if len(self.thickness) != self.nlay:
            raise ValueError("Thickness array must have length nlay")

    # Darcy-based transmissivity (USBR §5–4, Todd 2005)
    def transmissivity_x(self, k, i, j):
        return self.Kx[k, i, j] * self.thickness[k]

    def transmissivity_y(self, k, i, j):
        return self.Ky[k, i, j] * self.thickness[k]
