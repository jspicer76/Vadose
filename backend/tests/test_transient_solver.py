# backend/tests/test_transient_solver.py

import numpy as np

from backend.solvers.transient.transient_solver import (
    TransientSolver,
    run_transient_solver,
)
from backend.solvers.transient.time_stepper import (
    FixedTimeStepper,
    ImplicitEulerStepper,
)
from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC
from backend.models.transient_model import ObservationPoint


class DummyModel:
    """Minimal model object needed to run transient solver."""

    def __init__(self, nx, ny, dx, dy, S, T):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        # Uniform transmissivity
        self.transmissivity = np.full((nx, ny), T)

        # Confined aquifer
        self.aquifer_type = "confined"
        self.S = S  # storativity

        # Active array
        self.active = np.ones((nx, ny), dtype=bool)

        # Initial heads
        self.h0 = np.zeros((nx, ny))

        # Empty wells & recharge for now
        self.wells = []
        self.recharge = None

        # BC list
        self.boundary_conditions = []
        self.observation_points = []

    def get_recharge_field(self, _t):
        return self.recharge


def test_transient_diffusion():
    # Domain and properties
    nx, ny = 51, 1
    dx = dy = 2.0  # m
    S = 1e-4
    T = 1e-3

    model = DummyModel(nx, ny, dx, dy, S, T)

    # Left boundary fixed at 10 m
    left_cells = [0]  # cell (i=0, j=0)
    model.boundary_conditions.append(
        DirichletBC(cells=left_cells, head_value=10.0)
    )

    # Right boundary fixed at 0 m
    right_cells = [(nx - 1) * ny]  # last cell
    model.boundary_conditions.append(
        DirichletBC(cells=right_cells, head_value=0.0)
    )

    # Solver setup
    t_start = 0.0
    t_end = 5000.0
    dt = 50.0
    stepper = FixedTimeStepper(t_start, t_end, dt)
    integrator = ImplicitEulerStepper()
    solver = TransientSolver(model, stepper, integrator)

    # Track mid-domain observation
    obs_cell = (nx // 2, 0)
    model.observation_points = [
        ObservationPoint(obs_cell[0], obs_cell[1], "mid", model.h0[obs_cell])
    ]

    heads, obs_data, logs = solver.run(t_start, t_end, dt)

    # --- Tests ---

    # Head at left boundary should remain 10
    assert abs(heads[-1, 0, 0] - 10.0) < 1e-6

    # Head at right boundary should remain 0
    assert abs(heads[-1, -1, 0] - 0.0) < 1e-6

    # Interior head must be > 0 (diffusion)
    assert np.all(heads[-1, 1:-1, 0] > 0)

    # Head should increase monotonically at mid-domain
    mid = nx // 2
    mid_series = heads[:, mid, 0]
    assert np.all(np.diff(mid_series) >= -1e-6)

    obs = obs_data
    assert obs["_times"].shape[0] == heads.shape[0]
    assert "mid" in obs["head"]
    assert np.allclose(obs["head"]["mid"], heads[:, mid, 0])
    assert np.all(obs["drawdown"]["mid"] <= 0.0)

    print("\n=== TRANSIENT DIFFUSION TEST PASSED ===")


def test_run_helper_observation_payload():
    config = {
        "nx": 2,
        "ny": 2,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 10.0,
        "steps": 2,
        "T": 5.0,
        "S": 0.2,
        "initial_head": 10.0,
        "recharge": 1e-6,
        "observation_points": [{"name": "corner", "i": 0, "j": 0}],
        "return_full_output": True,
    }

    result = run_transient_solver(config)
    heads = result["heads"]
    obs = result["observations"]

    assert heads.shape[0] == len(obs["_times"])
    assert "corner" in obs["head"]
    assert obs["head"]["corner"].shape[0] == len(obs["_times"])
    # Drawdown should remain small and non-positive relative to initial head
    assert np.all(obs["drawdown"]["corner"] <= 0.0 + 1e-9)
