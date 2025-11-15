# backend/solvers/transient/matrix_assembly.py

import numpy as np

class MatrixAssembly:
    """
    Builds finite-difference conductance matrix A (NxN)
    using row-major indexing k = i*ny + j.

    Supports:
    - Confined or unconfined transmissivity
    - Harmonic averaging between neighbors
    - No-flow boundaries (inactive neighbors)
    """

    @staticmethod
    def build_conductance_matrix(model):
        nx, ny = model.nx, model.ny
        N = nx * ny

        dx = model.dx
        dy = model.dy

        # In MODFLOW, transmissivity is given per cell.
        T = model.transmissivity  # shape (nx, ny)

        # Inactive cell mask
        active = model.active  # boolean array (nx, ny)

        A = np.zeros((N, N))

        # Precompute reciprocals
        dx2 = dx * dx
        dy2 = dy * dy

        # Loop through all active cells
        for i in range(nx):
            for j in range(ny):

                if not active[i, j]:
                    continue  # skip inactive

                k = i * ny + j

                # Conductance accumulator for diagonal
                diag = 0.0

                # -----------------------------
                # WEST neighbor (i-1, j)
                # -----------------------------
                if i > 0 and active[i-1, j]:
                    Tw = T[i, j]
                    Te = T[i-1, j]
                    Cw = (2 * Tw * Te) / (Tw + Te) / dx2

                    kw = (i-1) * ny + j

                    A[k, kw] -= Cw
                    diag += Cw

                # -----------------------------
                # EAST neighbor (i+1, j)
                # -----------------------------
                if i < nx - 1 and active[i+1, j]:
                    Tw = T[i, j]
                    Te = T[i+1, j]
                    Ce = (2 * Tw * Te) / (Tw + Te) / dx2

                    ke = (i+1) * ny + j

                    A[k, ke] -= Ce
                    diag += Ce

                # -----------------------------
                # SOUTH neighbor (i, j-1)
                # -----------------------------
                if j > 0 and active[i, j-1]:
                    Tn = T[i, j]
                    Ts = T[i, j-1]
                    Cs = (2 * Tn * Ts) / (Tn + Ts) / dy2

                    ks = i * ny + (j-1)

                    A[k, ks] -= Cs
                    diag += Cs

                # -----------------------------
                # NORTH neighbor (i, j+1)
                # -----------------------------
                if j < ny - 1 and active[i, j+1]:
                    Tn = T[i, j]
                    Ts = T[i, j+1]
                    Cn = (2 * Tn * Ts) / (Tn + Ts) / dy2

                    kn = i * ny + (j+1)

                    A[k, kn] -= Cn
                    diag += Cn

                # -----------------------------
                # Diagonal entry
                # -----------------------------
                A[k, k] = diag

        return A
