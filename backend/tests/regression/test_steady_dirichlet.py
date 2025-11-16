import numpy as np
from backend.tests.regression.helpers import linear_head_profile
from backend.solvers.transient.transient_solver import run_transient_solver

def test_steady_dirichlet():
    nx = 21
    left_h = 100.0
    right_h = 90.0
    
    config = {
        "nx": nx,
        "ny": nx,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 1.0,
        "steps": 1,
        "T": 1.0,
        "S": 0.0,
        "initial_head": 95.0,
        "boundary_conditions": [
            {"type": "dirichlet", "cells": [(i,0) for i in range(nx)], "value": left_h},
            {"type": "dirichlet", "cells": [(i,nx-1) for i in range(nx)], "value": right_h},
        ],
    }
    
    num = run_transient_solver(config)[-1]
    ana = linear_head_profile(nx, left_h, right_h)
    
    # Compare center row where BCs aren't applied
    row = nx // 2
    assert np.max(np.abs(num[row, :] - ana)) < 1e-6
