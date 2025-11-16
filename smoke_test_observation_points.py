# smoke_test_observation_points.py

import numpy as np

from backend.models.transient_model import TransientModel
from backend.solvers.transient.time_stepper import FixedTimeStepper, ImplicitEulerStepper
from backend.solvers.transient.transient_solver import TransientSolver

# --------------------------------------------------------------
# Grid and aquifer properties
# --------------------------------------------------------------
nx = ny = 3
dx = dy = 10.0

T = 1000.0
S = 1e-4
h0 = 10.0
recharge = 1e-8   # small realistic recharge


# --------------------------------------------------------------
# Build model
# --------------------------------------------------------------
model = TransientModel(
    nx=nx, ny=ny,
    dx=dx, dy=dy,
    T=T, S=S,
    h0=h0
)

model.b = 10.0             # saturated thickness
model.recharge = recharge  # uniform recharge

# Add observation points
model.add_observation_point(1, 1, name="center")
model.add_observation_point(0, 0, name="corner")


# --------------------------------------------------------------
# Time stepping: 6 hours in 5-minute steps
# --------------------------------------------------------------
dt = 300  # seconds
t_start = 0
t_end = 6 * 3600

stepper = FixedTimeStepper(t_start, t_end, dt)
integrator = ImplicitEulerStepper()

solver = TransientSolver(model, stepper, integrator)

# --------------------------------------------------------------
# Run the simulation
# --------------------------------------------------------------
heads, obs_data, logs = solver.run(t_start, t_end, dt)

print("\n=== Observation Time Series (Step 3 Test) ===")
print("Times (s):", obs_data["_times"])
print("\nCenter Heads:", obs_data["center"])
print("Corner Heads:", obs_data["corner"])

print("\nFinal head field:")
print(heads[-1])

print("\nDrawdown at center:", obs_data["center"][-1] - obs_data["center"][0])
