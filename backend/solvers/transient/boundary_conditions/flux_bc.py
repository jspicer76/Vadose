# backend/solvers/transient/boundary_conditions/flux_bc.py

import numpy as np

class FluxBC:
    """
    Neumann (specified flux) boundary condition.

    Positive flux  = inflow  to aquifer
    Negative flux  = outflow from aquifer

    Flux is given in L/T (m/s), and is converted to volumetric rate
    by multiplying by cell face area (dx or dy depending on orientation).
    """

    def __init__(self, cells, flux, model):
        """
        cells: list of linear indices into solver matrix
        flux: scalar, array, or callable f(t)
        model: TransientModel instance (needed for dx/dy)
        """
        self.cells = cells
        self.model = model
        self.flux = flux  # [L/T]

        # Precompute face area assuming boundary faces are one cell wide.
        # MODFLOW treats flux BCs as applied over the cell face perpendicular
        # to the flow direction. For simplicity, we assume the boundary face
        # is dx*1 or dy*1 depending on model geometry.
        self.face_area = self.model.dx * self.model.dy  # treat as 2D volumetric cell-wide flux

    def get_flux_volume(self, t):
        """
        Returns volumetric rate [L^3/T] for each BC cell.
        """
        nx, ny = self.model.nx, self.model.ny

        # Evaluate flux — may be scalar, array, or callable.
        if callable(self.flux):
            f = self.flux(t)
        else:
            f = self.flux

        # Convert scalar → per-cell value
        if np.isscalar(f):
            f = np.full(len(self.cells), f, dtype=float)
        else:
            f = np.asarray(f, dtype=float)

        # volumetric = flux * face_area
        return f * self.face_area

    def apply_to_rhs(self, rhs, t):
        """
        Apply flux BC to the RHS vector.
        rhs: flattened RHS vector.
        """
        Q = self.get_flux_volume(t)  # volumetric [L^3/T]
        for idx, cell in enumerate(self.cells):
            rhs[cell] += Q[idx]
        return rhs
