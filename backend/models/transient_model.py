# backend/models/transient_model.py

import numpy as np

class Well:
    """Simple well wrapper for compatibility with SourceSink."""
    def __init__(self, i, j, Q):
        self.cell = (i, j)
        self.Q = Q


class TransientModel:
    """
    Minimal model class for the transient groundwater solver.
    """

    def __init__(
        self,
        nx, ny,
        dx, dy,
        T, S,
        h0,
        pumping_wells=None,
        boundary_conditions=None
    ):
        # Grid geometry
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        # Hydraulic properties (scalar)
        self.T = T          # transmissivity
        self.S = S          # storativity
        self.aquifer_type = "confined"   # REQUIRED by Storage.compute_storage()

        # Initial head field
        # Ensure initial head is a full (nx, ny) grid
        if np.isscalar(h0):
            self.h0 = np.full((nx, ny), float(h0), dtype=float)
        else:
            arr = np.asarray(h0, dtype=float)
            if arr.shape != (nx, ny):
                arr = arr.reshape((nx, ny))
            self.h0 = arr




        # Active cells (all active)
        self.active = np.ones((nx, ny), dtype=bool)

        # Wells — convert dicts to Well objects
        self.pumping_wells = []
        if pumping_wells:
            for w in pumping_wells:
                self.pumping_wells.append(Well(w["i"], w["j"], w["Q"]))

        # Boundary conditions
        self.boundary_conditions = boundary_conditions or []

        # Storage term (updated every step)
        self.storage = None

        # Budget print interval
        self.budget_interval = 10

    # -------------------------------------------------------------------
    # Required properties for solver stack
    # -------------------------------------------------------------------

    @property
    def transmissivity(self):
        """Full 2D transmissivity field."""
        return np.full((self.nx, self.ny), self.T, dtype=float)

    @property
    def storativity(self):
        """Scalar storativity."""
        return self.S

    @property
    def wells(self):
        """Return list of Well objects."""
        return self.pumping_wells
