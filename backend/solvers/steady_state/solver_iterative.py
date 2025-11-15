import numpy as np

def solve_iterative(A, b, nx, ny, w=1.4, max_iter=3000, tol=1e-6, verbose=False):
    """
    Solve Ah = b using Gauss–Seidel or SOR iteration.

    Parameters:
        A : sparse CSR matrix
        b : RHS vector (flattened, length = nx*ny)
        nx, ny : grid dimensions
        w : relaxation factor (1.0 = Gauss–Seidel, 1.0 < w < 2.0 = SOR)
        max_iter : maximum inner iterations
        tol : convergence tolerance on residual norm
        verbose : print iteration info

    Returns:
        head : 2D array (nx, ny)
        iters : number of iterations performed
        converged : bool
    """

    N = nx * ny
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr

    # Initial guess for head
    h = np.zeros(N)

    # Gauss–Seidel / SOR iteration
    for it in range(max_iter):

        h_old = h.copy()

        # Loop over rows of the matrix
        for row in range(N):
            diag = None
            sum_neighbors = 0.0

            start = A_indptr[row]
            end = A_indptr[row + 1]

            for idx in range(start, end):
                col = A_indices[idx]
                val = A_data[idx]

                if col == row:
                    diag = val
                else:
                    sum_neighbors += val * h[col]

            if diag is None or abs(diag) < 1e-20:
                continue  # skip singular rows

            # Gauss–Seidel update (inner)
            h_gs = (b[row] - sum_neighbors) / diag

            # SOR update
            h[row] = (1 - w) * h[row] + w * h_gs

        # Convergence check
        residual = np.linalg.norm(h - h_old, ord=np.inf)
        if verbose and (it % 100 == 0):
            print(f"Iter {it:4d}: Residual = {residual:.6e}")

        if residual < tol:
            return h.reshape((ny, nx)).T, it + 1, True

    # If reached here → did not converge
    return h.reshape((ny, nx)).T, max_iter, False
