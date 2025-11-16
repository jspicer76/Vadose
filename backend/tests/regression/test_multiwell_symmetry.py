import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_multiwell_symmetry():
    nx = 21
    center = nx // 2
    
    wells = [
        {"i": center, "j": center-2, "Q": 0.01},
        {"i": center, "j": center+2, "Q": 0.01},
    ]

    config = {
        "nx": nx, "ny": nx,
        "dx": 10, "dy": 10,
        "dt": 5,  "steps": 50,
        "T": 100.0, "S": 0.001,
        "initial_head": 10.0,
        "pumping_wells": wells,
    }

    h = run_transient_solver(config)[-1]

    left = h[center, center-5]
    right = h[center, center+5]

    assert abs(left - right) < 1e-6
