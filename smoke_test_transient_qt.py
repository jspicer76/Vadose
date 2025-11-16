# smoke_test_transient_qt.py

import numpy as np
from backend.solvers.transient.transient_solver import TransientSolver
from backend.time_steppers.backward_euler import BackwardEuler
from backend.models.transient_model import TransientModel, Well

print(">>> ENTERING SMOKE TEST FILE <<<")

# --------------------------------------------
# Minimal 1x1 grid for testing
# --------------------------------------------
class FakeGrid:
    def __init__(self):
        self.nrow = 1
        self.ncol = 1
        self.num_cells = 1

    def cell_index(self, r, c):
        return 0


# --------------------------------------------
# Fake model that bypasses real constructor
# but satisfies ALL attributes the solver needs
# --------------------------------------------
class FakeTransientModel(TransientModel):
    def __init__(self):
        # 1. Grid
        self.nx = 1
        self.ny = 1
        self.dx = 1.0
        self.dy = 1.0
        self.N = 1
        self.grid = FakeGrid()

        # 2. Initial head
        self.h0 = np.array([[100.0]])
        self.initial_head = np.array([100.0])  # 1D for solver

        # 3. Aquifer properties
        self.T = 1.0      # transmissivity (scalar)
        self.S = 1.0      # storativity (scalar)
        self.aquifer_type = "confined"

        # 4. Active mask
        self.active = np.ones((1, 1), dtype=bool)

        # 5. Wells (must be Well objects!)
        w = Well(0, 0, -500.0)
        w.schedule = {
            "type": "step",
            "times": [0.0, 5.0],
            "rates": [-500.0, -1000.0]
        }
        self.pumping_wells = [w]

        # 6. Recharge & boundaries
        self.boundary_conditions = []
        self.recharge = None

        # 7. Observation points
        self.observation_points = []

        # 8. Additional required fields
        self.layers = 1
        self.storage = None
        self.budget_interval = 10


# --------------------------------------------
# Run the transient solver
# --------------------------------------------
print(">>> BEFORE MODEL BUILD <<<")
model = FakeTransientModel()

print(">>> BEFORE SOLVER RUN <<<")
solver = TransientSolver(model, BackwardEuler())
heads, obs, logs = solver.run(t_start=0, t_end=10, dt=1)

print(">>> AFTER SOLVER RUN <<<")

# --------------------------------------------
# Print results
# --------------------------------------------
print("\nSMOKE TEST COMPLETED.")
print("HEADS:")
for t_index, h in enumerate(heads):
    print(f"t = {t_index}: {h}")
