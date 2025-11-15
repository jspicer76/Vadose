import numpy as np
from scipy.sparse.linalg import spsolve


def solve_direct(A, b, nx, ny):
    """
    Solve the linear system A h = b using a direct sparse solver (spsolve).

    Parameters:
        A : sparse CSR matrix
        b : RHS vector (length nx*ny)
        nx, ny : grid dimensions

    Returns:
        head : 2D array (nx, ny)
    """

    # Solve
    h_flat = spsolve(A, b)

    # Reshape into (nx, ny)
    head = h_flat.reshape((ny, nx)).T

    return head
