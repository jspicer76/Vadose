# backend/tests/test_transient_solver.py

import numpy as np

from backend.solvers.transient.transient_solver import TransientSolver
from backend.solvers.transient.time_stepper import ImplicitEulerStepper
from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC


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
    stepper = ImplicitEulerStepper()
    solver = TransientSolver(model, stepper)

    t_start = 0.0
    t_end = 5000.0
    dt = 50.0

    heads, logs = solver.run(t_start, t_end, dt)

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

    print("\n=== TRANSIENT DIFFUSION TEST PASSED ===")
