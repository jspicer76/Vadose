# backend/solvers/transient/matrix_assembly.py

import numpy as np

class MatrixAssembly:
    """
    Conductance matrix builder for transient groundwater flow.

    New architecture:
      - Accepts model in constructor
      - Provides a .matrix() method returning A (conductance matrix)
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    def matrix(self):
        """Build and return the conductance matrix A."""
        model = self.model
        nx, ny = model.nx, model.ny
        dx, dy = model.dx, model.dy

        # Accept transmissivity as either T or transmissivity
        if hasattr(model, "transmissivity"):
            T = model.transmissivity
        else:
            T = model.T

        # Accept active mask or default to all active
        active = getattr(model, "active", np.ones((nx, ny), dtype=bool))

        A = np.zeros((nx * ny, nx * ny), dtype=float)

        dx2 = dx * dx
        dy2 = dy * dy

        for i in range(nx):
            for j in range(ny):

                if not active[i, j]:
                    continue

                k = i * ny + j  # flat index
                diag = 0.0

                # WEST (i-1)
                if i > 0 and active[i - 1, j]:
                    Tw, Te = T[i, j], T[i - 1, j]
                    Cw = (2 * Tw * Te) / (Tw + Te) / dx2
                    kw = (i - 1) * ny + j
                    A[k, kw] -= Cw
                    diag += Cw

                # EAST (i+1)
                if i < nx - 1 and active[i + 1, j]:
                    Tw, Te = T[i, j], T[i + 1, j]
                    Ce = (2 * Tw * Te) / (Tw + Te) / dx2
                    ke = (i + 1) * ny + j
                    A[k, ke] -= Ce
                    diag += Ce

                # SOUTH (j-1)
                if j > 0 and active[i, j - 1]:
                    Tn, Ts = T[i, j], T[i, j - 1]
                    Cs = (2 * Tn * Ts) / (Tn + Ts) / dy2
                    ks = i * ny + (j - 1)
                    A[k, ks] -= Cs
                    diag += Cs

                # NORTH (j+1)
                if j < ny - 1 and active[i, j + 1]:
                    Tn, Ts = T[i, j], T[i, j + 1]
                    Cn = (2 * Tn * Ts) / (Tn + Ts) / dy2
                    kn = i * ny + (j + 1)
                    A[k, kn] -= Cn
                    diag += Cn

                # Diagonal
                A[k, k] = diag

        return A
