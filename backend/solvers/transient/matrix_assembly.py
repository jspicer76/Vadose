# backend/solvers/transient/matrix_assembly.py

import numpy as np

class MatrixAssembly:
    @staticmethod
    def build_conductance_matrix(model):
        nx, ny = model.nx, model.ny
        dx, dy = model.dx, model.dy

        T = model.transmissivity
        active = model.active

        A = np.zeros((nx * ny, nx * ny), dtype=float)

        dx2 = dx * dx
        dy2 = dy * dy

        for i in range(nx):
            for j in range(ny):

                if not active[i, j]:
                    continue

                k = i * ny + j
                diag = 0.0

                # WEST
                if i > 0 and active[i-1, j]:
                    Tw, Te = T[i, j], T[i-1, j]
                    Cw = (2 * Tw * Te) / (Tw + Te) / dx2
                    kw = (i-1) * ny + j
                    A[k, kw] -= Cw
                    diag += Cw

                # EAST
                if i < nx - 1 and active[i+1, j]:
                    Tw, Te = T[i, j], T[i+1, j]
                    Ce = (2 * Tw * Te) / (Tw + Te) / dx2
                    ke = (i+1) * ny + j
                    A[k, ke] -= Ce
                    diag += Ce

                # SOUTH
                if j > 0 and active[i, j-1]:
                    Tn, Ts = T[i, j], T[i, j-1]
                    Cs = (2 * Tn * Ts) / (Tn + Ts) / dy2
                    ks = i * ny + (j-1)
                    A[k, ks] -= Cs
                    diag += Cs

                # NORTH
                if j < ny - 1 and active[i, j+1]:
                    Tn, Ts = T[i, j], T[i, j+1]
                    Cn = (2 * Tn * Ts) / (Tn + Ts) / dy2
                    kn = i * ny + (j+1)
                    A[k, kn] -= Cn
                    diag += Cn

                A[k, k] = diag

        return A
