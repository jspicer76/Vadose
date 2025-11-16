import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_grid_anisotropy_stability():
    nx = 11
    config = {
        "nx": nx, "ny": nx,
        "dx": 10.0, "dy": 2.0,
        "dt": 10.0, "steps": 20,
        "T": 100.0, "S": 0.001,
        "initial_head": 10.0,
        "pumping_wells": [{"i":5, "j":5, "Q":0.01}],
    }
    h = run_transient_solver(config)[-1]
    assert not np.isnan(h).any()
