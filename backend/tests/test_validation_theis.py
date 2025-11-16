"""
Test: Numerical Solver vs Analytical Theis Solution
===================================================

Compares transient numerical groundwater drawdown to the classical
Theis (1935) analytical solution.

References:
- Theis (1935) nonequilibrium method.
- Todd & Mays (2005), Chapter 5.
- USBR Ground Water Manual (1995), Chapter IX.

Validation approach:
--------------------
1. Simulate pumping in a confined aquifer using numerical solver.
2. Compute analytical Theis drawdown for same distances and times.
3. Compute RMSE and ensure error remains below tolerance.
"""

import numpy as np
import pytest

from backend.validation.theis import theis_drawdown
from backend.solvers.transient.transient_solver import run_transient_solver



def radial_distance(i, j, center_i, center_j, dx):
    """Compute radial distance (meters) from well cell center."""
    return np.sqrt(((i - center_i) * dx) ** 2 + ((j - center_j) * dx) ** 2)


@pytest.mark.parametrize("Q,T,S", [
    (0.005, 100.0, 0.001),      # Small pumping rate, transmissive aquifer
])
def test_theis_validation(Q, T, S):

    # --------------------------------------------------------------
    # 1. Define model grid
    # --------------------------------------------------------------
    nx = ny = 21
    dx = dy = 10.0   # 10 m cell size
    center = (nx // 2, ny // 2)

    # --------------------------------------------------------------
    # 2. Numerical transient solver run
    # --------------------------------------------------------------
    model_config = {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "dt": 10.0,           # 10-second timestep
        "steps": 40,          # 400 seconds total
        "T": T,
        "S": S,
        "pumping_wells": [
            {"i": center[0], "j": center[1], "Q": Q}
        ],
        "initial_head": 10.0,
    }

    # Run numerical model
    numerical_heads = run_transient_solver(model_config)

    # Extract final head field after all timesteps
    num_final = numerical_heads[-1]
    h0 = model_config["initial_head"]
    num_drawdown = h0 - num_final

    # --------------------------------------------------------------
    # 3. Compute analytical drawdown (Theis)
    # --------------------------------------------------------------
    t_final = model_config["dt"] * model_config["steps"]

    ana_drawdown = np.zeros_like(num_drawdown)

    for i in range(nx):
        for j in range(ny):
            r = radial_distance(i, j, center[0], center[1], dx)
            if r < dx / 2:
                # Theis is undefined at r = 0; skip the center well cell
                ana_drawdown[i, j] = np.nan
            else:
                ana_drawdown[i, j] = theis_drawdown(Q, T, S, r, t_final)

    # Filter out invalid (NaN) cells and the well block
    valid_mask = ~np.isnan(ana_drawdown)
    ana_valid = ana_drawdown[valid_mask]
    num_valid = num_drawdown[valid_mask]

    # --------------------------------------------------------------
    # 4. Compute error metrics
    # --------------------------------------------------------------
    rmse = np.sqrt(np.mean((ana_valid - num_valid) ** 2))
    max_draw = np.nanmax(ana_valid)
    tolerance = 0.05 * max_draw   # 5% of max Theis drawdown

    print("\n--- THEIS VALIDATION RESULTS ---")
    print(f"RMSE:        {rmse:.6f}")
    print(f"Max draw:    {max_draw:.6f}")
    print(f"Tolerance:   {tolerance:.6f}")
    print("--------------------------------")

    assert rmse < tolerance, "Numerical solver deviates too far from Theis analytical solution."
