import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_uniform_recharge_small_dt():
    nx = 21
    R = 1e-6     # recharge m/s
    dt = 10.0
    S = 0.1

    expected_dh = (R * dt) / S

    config = {
        "nx": nx,
        "ny": nx,
        "dx": 1.0,
        "dy": 1.0,
        "dt": dt,
        "steps": 1,
        "T": 1.0,
        "S": S,
        "initial_head": 0.0,
        "recharge": np.full((nx,nx), R),
    }

    num = run_transient_solver(config)[-1]
    diff = num.mean()
    assert abs(diff - expected_dh) < 1e-6
