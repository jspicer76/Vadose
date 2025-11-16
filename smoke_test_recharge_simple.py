import numpy as np
from backend.solvers.transient.transient_solver import TransientSolver
from backend.time_steppers.backward_euler import BackwardEuler
from smoke_test_transient_qt import FakeGrid, FakeTransientModel

class RModel(FakeTransientModel):
    def __init__(self):
        super().__init__()
        self.nx = 1
        self.ny = 1
        self.dx = 1.0
        self.dy = 1.0
        self.grid = FakeGrid()

        self.h0 = np.array([[100.0]])
        self.initial_head = self.h0.flatten()

        # Remove the pumping well
        self.pumping_wells = []

        # Add constant recharge: 1e-6 m/s
        self.recharge = 1e-6

model = RModel()
solver = TransientSolver(model, BackwardEuler())

heads, obs, logs = solver.run(0, 5, 1)

print("HEADS:")
for t, h in enumerate(heads):
    print(t, h)
