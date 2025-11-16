import numpy as np

from backend.models.transient_model import TransientModel
from backend.solvers.transient.transient_solver import TransientSolver
from backend.solvers.transient.time_stepper import (
    FixedTimeStepper,
    ImplicitEulerStepper,
)


# ------------------------------------------------------
# 1. Build a very small 3×3 transient aquifer
# ------------------------------------------------------
nx, ny = 3, 3
dx = dy = 10.0

T = 1000.0       # transmissivity
S = 1e-4         # storativity
h0 = 10.0       # starting head everywhere

# Recharge: 1e-5 m/s applied uniformly
recharge_rate = 1e-7


# ------------------------------------------------------
# 2. Build transient model
# ------------------------------------------------------
model = TransientModel(
    nx=nx, ny=ny,
    dx=dx, dy=dy,
    T=T, S=S,
    h0=h0,
    pumping_wells=None,
    boundary_conditions=None,
    recharge=recharge_rate,
)


# ------------------------------------------------------
# 3. Time settings: 6 hours in 1-hour steps
# ------------------------------------------------------
t_start = 0
t_end   = 6 * 3600
dt      = 300       # 1 hour

stepper = FixedTimeStepper(t_start, t_end, dt)
integrator = ImplicitEulerStepper()
solver = TransientSolver(model, stepper, integrator)


# ------------------------------------------------------
# 4. Run solver
# ------------------------------------------------------
heads, logs = solver.run(t_start, t_end, dt)
final_heads = heads[-1]
print("\n=== Final Heads After Recharge ===")
print(final_heads)

# Quick sanity: heads should increase
print("\nHead increase (m):")
print(final_heads - model.h0)
