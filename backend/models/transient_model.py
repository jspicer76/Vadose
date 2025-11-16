# backend/models/transient_model.py

import numpy as np


class Well:
    """
    Simple well wrapper for compatibility with SourceSink.
    Each well has:
        (i, j): cell index
        Q     : discharge (positive = injection, negative = pumping)
    """
    def __init__(self, i, j, Q):
        self.cell = (i, j)
        self.Q = Q


class TransientModel:
    """
    Minimal model class for the transient groundwater solver
    with recharge support.

    Handles:
      - grid geometry
      - hydraulic properties (T, S)
      - initial heads
      - wells
      - boundary conditions
      - recharge (scalar, array, or callable f(t))
    """

    def __init__(
        self,
        nx, ny,
        dx, dy,
        T, S,
        h0,
        pumping_wells=None,
        boundary_conditions=None,
        recharge=None,              # NEW
    ):
        # --------------------------------------------------------------
        # Grid geometry
        # --------------------------------------------------------------
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.N = nx * ny            # total cells (flattened indexing)

        # --------------------------------------------------------------
        # Aquifer hydraulic properties
        # --------------------------------------------------------------
        self.T = T                  # transmissivity (scalar for now)
        self.S = S                  # storativity (scalar)
        self.aquifer_type = "confined"   # required by Storage.compute_storage()

        # --------------------------------------------------------------
        # Initial head field h0
        # Accept scalar, 2D array, or flat array
        # --------------------------------------------------------------
        arr = np.asarray(h0, dtype=float)

        if np.isscalar(h0) or arr.size == 1:
            self.h0 = np.full((nx, ny), float(arr), dtype=float)
        elif arr.shape == (nx, ny):
            self.h0 = arr.copy()
        elif arr.size == nx * ny:
            self.h0 = arr.reshape((nx, ny))
        else:
            raise ValueError(
                f"Initial head has shape {arr.shape}, expected {(nx, ny)}"
            )

        # --------------------------------------------------------------
        # Active cells — simple binary mask for now
        # --------------------------------------------------------------
        self.active = np.ones((nx, ny), dtype=bool)

        # --------------------------------------------------------------
        # Wells — convert dicts to Well objects
        # --------------------------------------------------------------
        self.pumping_wells = []
        if pumping_wells:
            for w in pumping_wells:
                self.pumping_wells.append(Well(w["i"], w["j"], w["Q"]))

        # --------------------------------------------------------------
        # Boundary conditions
        # --------------------------------------------------------------
        self.boundary_conditions = boundary_conditions or []

        # --------------------------------------------------------------
        # Recharge (scalar, 2D array, or callable f(t))
        # --------------------------------------------------------------
        self.recharge = recharge

        # Used by solver during timestepping
        self.storage = None
        self.budget_interval = 10

    # ==============================================================
    # Helper: Fetch wells
    # ==============================================================
    @property
    def wells(self):
        return self.pumping_wells

    # ==============================================================
    # 2D Transmissivity field
    # ==============================================================
    @property
    def transmissivity(self):
        return np.full((self.nx, self.ny), self.T, dtype=float)

    # ==============================================================
    # Storativity field (scalar for now)
    # ==============================================================
    @property
    def storativity(self):
        return self.S

    # ==============================================================
    # Recharge interface (NEW)
    # ==============================================================
    def get_recharge_field(self, t):
        """
        Return recharge R(x,y) [L/T] at time t.

        Acceptable forms:
          - None
          - scalar
          - 2D array (nx, ny)
          - callable f(t) → scalar or array
        """
        if self.recharge is None:
            return None

        R = self.recharge

        # Callable: f(t)
        if callable(R):
            R = R(t)

        # Scalar → expand to full array
        if np.isscalar(R):
            return np.full((self.nx, self.ny), float(R), dtype=float)

        R = np.asarray(R, dtype=float)

        # Accept 2D matching grid
        if R.shape == (self.nx, self.ny):
            return R

        # Accept flat array of correct size
        if R.size == self.N:
            return R.reshape((self.nx, self.ny))

        raise ValueError(
            f"Recharge shape {R.shape} is invalid — "
            f"expected scalar, {(self.nx, self.ny)}, or size={self.N}."
        )
