# backend/solvers/transient/boundary_conditions/theis_dirichlet.py

import numpy as np
from backend.validation.theis import theis_drawdown


class TheisDirichletBC:
    """
    Time-varying Dirichlet boundary that enforces Theis drawdown
    at the model perimeter to mimic an infinite aquifer.
    """

    def __init__(self, cells, cell_coords, wells, T, S, dx, dy, head0):
        """
        Parameters
        ----------
        cells : list[int]
            Linear indices of boundary cells (row-major).
        cell_coords : list[tuple[int, int]]
            (i, j) indices corresponding to `cells`.
        wells : list[tuple[int, int, float]]
            Sequence of (i, j, Q) for pumping wells.
        T : float
            Transmissivity.
        S : float
            Storativity.
        dx, dy : float
            Grid spacing in x and y directions.
        head0 : float
            Initial head.
        """
        self.cells = cells
        self.cell_coords = cell_coords
        self.wells = wells
        self.T = T
        self.S = S
        self.dx = dx
        self.dy = dy
        self.head0 = head0
        self.current_heads = np.full(len(cells), head0, dtype=float)

    def update(self, t):
        """Update boundary heads at simulation time t (seconds)."""
        if t <= 0:
            self.current_heads.fill(self.head0)
            return

        for idx, (i, j) in enumerate(self.cell_coords):
            drawdown = 0.0
            for wi, wj, Q in self.wells:
                rx = (i - wi) * self.dx
                ry = (j - wj) * self.dy
                r = np.hypot(rx, ry)
                # Avoid r=0 singularities (should not occur on perimeter).
                r_eff = max(r, 0.5 * min(self.dx, self.dy))
                drawdown += theis_drawdown(abs(Q), self.T, self.S, r_eff, t)
            self.current_heads[idx] = self.head0 - drawdown

    def apply(self, A, b):
        """Apply Dirichlet condition using current head values."""
        for value, idx in zip(self.current_heads, self.cells):
            A[idx, :] = 0.0
            A[idx, idx] = 1.0
            b[idx] = value
        return A, b
