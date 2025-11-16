import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_asymmetric_bc_solution_shape():
    nx = 21
    left  = 100.0
    right = 95.0

    config = {
        "nx": nx, "ny": nx,
        "dx": 10, "dy": 10,
        "dt": 1, "steps": 1,
        "T": 1.0, "S": 0.0,
        "initial_head": 97.0,
        "boundary_conditions": [
            {"type": "dirichlet", "cells": [(i,0) for i in range(nx)], "value": left},
            {"type": "dirichlet", "cells": [(i,nx-1) for i in range(nx)], "value": right},
        ],
    }

    h = run_transient_solver(config)[-1]
    assert h.shape == (nx,nx)
    assert abs(h[:,0].mean()  - left)  < 1e-6
    assert abs(h[:,-1].mean() - right) < 1e-6
