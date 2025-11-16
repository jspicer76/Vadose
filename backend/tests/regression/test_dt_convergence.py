import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_dt_halving_convergence():
    nx = 21
    base_config = {
        "nx": nx, "ny": nx,
        "dx": 10, "dy": 10,
        "T": 50.0, "S": 0.001,
        "initial_head": 10.0,
        "pumping_wells": [{"i": 10, "j": 10, "Q": 0.01}],
    }

    c1 = base_config.copy(); c1["dt"] = 10; c1["steps"] = 40
    c2 = base_config.copy(); c2["dt"] =  5; c2["steps"] = 80

    h1 = run_transient_solver(c1)[-1]
    h2 = run_transient_solver(c2)[-1]

    err = np.max(np.abs(h1 - h2))
    assert err < 1e-3
