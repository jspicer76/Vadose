import numpy as np

from .assemble_matrix import assemble_matrix
from .solver_direct import solve_direct
from .solver_iterative import solve_iterative


def solve_steady_state(
    model,
    method="direct",
    max_outer_iter=20,
    max_inner_iter=3000,
    tol_head=1e-4,
    tol_inner=1e-6,
    w=1.4,
    verbose=False
):
    """
    Unified solver for steady-state confined + unconfined groundwater flow.

    Outer loop:
        - updates saturated thickness (unconfined BCF method)
        - assembles matrix A and RHS vector b
        - solves Ah = b with selected method

    Supported methods:
        - "direct"   → sparse direct solver (spsolve)
        - "sor"      → Gauss–Seidel / SOR

    Parameters:
        model : AquiferModel
        method : "direct" or "sor"
        max_outer_iter : limit on unconfined outer iterations
        max_inner_iter : limit on linear solver iterations for SOR
        tol_head : tolerance for outer head convergence
        tol_inner : tolerance for inner solver iterations
        w : relaxation factor for SOR
        verbose : print solver output

    Returns:
        head : 2D array (nx, ny)
        outer_iters : number of outer iterations
        inner_info : data about inner solver (iterations, convergence)
        converged : bool
    """

    grid = model.grid
    nx, ny = grid.nx, grid.ny

    # initial guess — flat head field, set to first constant-head boundary if available
    head_prev = _initial_head_guess(model, nx, ny)

    for outer in range(max_outer_iter):

        # Assemble matrix with current saturated thickness from head_prev
        A, b = assemble_matrix(model, head_prev)

        # Select solver
        if method.lower() == "direct":
            head_new = solve_direct(A, b, nx, ny)
            inner_info = {"iters": 1, "converged": True}

        elif method.lower() in ("sor", "gs", "gauss-seidel"):
            head_new, iters, ok = solve_iterative(
                A, b, nx, ny, w=w,
                max_iter=max_inner_iter,
                tol=tol_inner,
                verbose=verbose
            )
            inner_info = {"iters": iters, "converged": ok}

        else:
            raise ValueError(f"Unknown solver method: {method}")

        # Outer loop convergence check (head stabilization)
        diff = np.max(np.abs(head_new - head_prev))

        if verbose:
            print(f"[Outer {outer+1}] Head change = {diff:.6e}")

        if diff < tol_head:
            return head_new, outer + 1, inner_info, True

        # Prepare for next iteration
        head_prev = head_new.copy()

    # Did not converge
    return head_new, max_outer_iter, inner_info, False


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _initial_head_guess(model, nx, ny):
    """
    Create an initial head field.
    1. If constant-head boundaries exist → use their average for initialization.
    2. Else → uniform head = 10.

    Returns:
        head[nx, ny]
    """
    head = np.ones((nx, ny)) * 10.0

    ch_values = [bc["value"] for bc in model.boundaries if bc["type"] == "CONSTANT_HEAD"]

    if len(ch_values) > 0:
        avg = sum(ch_values) / len(ch_values)
        head[:] = avg

    return head
