"""
Unified interface for steady-state groundwater flow solvers.

Provides:
 - Direct solver (LU)
 - SOR iterative solver
 - Unified result dictionary for AquiferModel and API

References:
  - Todd & Mays (2005) Groundwater Hydrology
  - USBR Ground Water Manual (1995)
"""

from __future__ import annotations
import numpy as np

from backend.solvers.steady_state.assemble_matrix import assemble_matrix
from backend.solvers.steady_state.solver_direct import solve_direct
from backend.solvers.steady_state.solver_iterative import solve_iterative
from backend.utils.logging import log_solver_start, log_solver_result


class SteadyStateSolver:
    """
    High-level steady-state solver wrapper.
    Supports: method = "direct" or "sor"
    """

    # ------------------------------------------------------------------
    #  MAIN SOLVER ENTRY POINT
    # ------------------------------------------------------------------
    def solve(self, model, method: str = "direct") -> dict:
        """
        Solve the steady-state saturated flow equation using
        a matrix-based finite-difference approach.

        Returns:
            {
                "converged": bool,
                "iterations": int | None,
                "method": str,
                "head": np.ndarray,
                "residual_norm": float,
                "notes": str
            }
        """

        method = method.lower()
        if method not in ("direct", "sor"):
            raise ValueError(f"Unknown solver method: {method}")

        log_solver_start(f"Running steady-state solver ({method})")

        # Initial head guess based on average constant-head BCs
        nx, ny = model.grid.nx, model.grid.ny
        head_prev = self._initial_head_guess(model, nx, ny)

        max_outer_iter = 20
        tol_outer = 1e-4

        iterations = None
        converged = False
        notes = ""
        residual = 0.0

        # ==================================================================
        #  OUTER LOOP — FOR UNCONFINED ITERATION (BCF METHOD)
        # ==================================================================
        for outer in range(max_outer_iter):

            # Assemble matrix using current head estimate
            A, b, grid_shape = assemble_matrix(model, head_prev)
            nx, ny = grid_shape

            # --------------------------------------------------------------
            # SELECT NUMERICAL SOLVER
            # --------------------------------------------------------------
            if method == "direct":
                # Direct solver returns a full (nx, ny) head array
                head_new = solve_direct(A, b, nx, ny)
                residual = 0.0
                iterations = None
                converged_inner = True
                notes = "Direct LU decomposition completed."

            else:  # SOR iterative solver
                # solve_iterative returns (head_2D, iterations, converged)
                head_new, iterations, converged_inner = solve_iterative(
                    A, b, nx, ny
                )

                # compute a residual for reporting
                residual = np.linalg.norm(A @ head_new.flatten() - b)

                notes = "SOR iteration completed."

            # --------------------------------------------------------------
            # OUTER LOOP CONVERGENCE CHECK (BCF method)
            # --------------------------------------------------------------
            diff = np.max(np.abs(head_new - head_prev))

            if diff < tol_outer:
                converged = True
                head_final = head_new
                break

            head_prev = head_new.copy()

        else:
            # OUTER LOOP DID NOT CONVERGE
            converged = False
            head_final = head_new

        # ==================================================================
        #  BUILD RETURN DICTIONARY
        # ==================================================================
        result = {
            "converged": bool(converged),
            "iterations": iterations,
            "method": method,
            "head": head_final,
            "residual_norm": float(residual),
            "notes": notes,
        }

        log_solver_result(result)
        return result

    # ------------------------------------------------------------------
    #  INITIAL GUESS FOR HEAD FIELD
    # ------------------------------------------------------------------
    def _initial_head_guess(self, model, nx, ny):
        """Initialize head with average constant-head boundary or 10 ft."""
        head = np.ones((nx, ny)) * 10.0

        ch_values = [
            bc["value"]
            for bc in model.boundaries
            if bc["type"] == "CONSTANT_HEAD"
        ]

        if ch_values:
            head[:] = sum(ch_values) / len(ch_values)

        return head


# ======================================================================
#  PUBLIC FUNCTION EXPECTED BY TEST SUITE
# ======================================================================

def solve_steady_state(model, method="direct"):
    """
    Simple wrapper function used by tests and API.
    """
    solver = SteadyStateSolver()
    return solver.solve(model, method)
