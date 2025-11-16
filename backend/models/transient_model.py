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


class ObservationPoint:
    """
    Stores metadata for an observation / monitoring location.
    """

    def __init__(self, i, j, name, reference_head):
        self.i = int(i)
        self.j = int(j)
        self.name = name
        self.reference_head = float(reference_head)

    @property
    def cell(self):
        return (self.i, self.j)


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
        observation_points=None,    # NEW: list of dicts/cells
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

        # --------------------------------------------------------------
        # Observation points (optional)
        # --------------------------------------------------------------
        self.observation_points = []
        if observation_points:
            self._init_observation_points(observation_points)

        # Used by solver during timestepping
        self.storage = None
        self.budget_interval = 10
    # --------------------------------------------------------------
    # Observation Points (Step 3)
    # --------------------------------------------------------------
    def add_observation_point(self, i, j, name=None):
        """
        Register an observation point at (i, j).
        This stores the point so the solver can record head(t) later.
        """
        if not hasattr(self, "observation_points"):
            self.observation_points = []

        if name is None:
            name = f"obs_{len(self.observation_points) + 1}"

        self.observation_points.append({
            "i": int(i),
            "j": int(j),
            "name": name
        })

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

    # ==============================================================
    # Observation helpers
    # ==============================================================
    def _init_observation_points(self, configs):
        for idx, conf in enumerate(configs):
            point = self._make_observation_point(conf, idx)
            self.observation_points.append(point)

    def _make_observation_point(self, conf, idx):
        if isinstance(conf, ObservationPoint):
            i, j = conf.cell
            name = conf.name or f"Obs-{idx+1}"
            ref = conf.reference_head
            return ObservationPoint(i, j, name, ref)

        if isinstance(conf, dict):
            if "cell" in conf:
                i, j = conf["cell"]
            else:
                i = conf.get("i")
                j = conf.get("j")
            if i is None or j is None:
                raise ValueError("Observation dict requires 'i' and 'j' or 'cell'")
            name = conf.get("name") or conf.get("label")
            ref = conf.get("reference_head")
        elif isinstance(conf, (tuple, list)) and len(conf) == 2:
            i, j = conf
            name = None
            ref = None
        else:
            raise ValueError(f"Invalid observation point definition: {conf}")

        i = int(i)
        j = int(j)
        self._validate_cell_indices(i, j)

        if ref is None:
            ref = float(self.h0[i, j])

        label = name or f"Obs-{idx+1}"
        return ObservationPoint(i, j, label, ref)

    def _validate_cell_indices(self, i, j):
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise ValueError(
                f"Observation cell {(i, j)} outside grid {(self.nx, self.ny)}"
            )

    def add_observation_point(self, i, j, name=None, reference_head=None):
        """
        Convenience helper to add an observation point after model creation.
        """
        conf = {"i": i, "j": j}
        if name is not None:
            conf["name"] = name
        if reference_head is not None:
            conf["reference_head"] = reference_head
        point = self._make_observation_point(conf, len(self.observation_points))
        self.observation_points.append(point)
        return point
