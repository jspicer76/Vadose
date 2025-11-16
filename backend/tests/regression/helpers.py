import numpy as np

def linear_head_profile(nx, left_h, right_h):
    """Analytical steady-state solution h(x) = left + (x/L)*(right-left)."""
    x = np.linspace(0, 1, nx)
    return left_h + x * (right_h - left_h)

def nearly_equal(a, b, tol=1e-6):
    return np.max(np.abs(a - b)) < tol

def run_simple_solver(run_transient_solver, nx, left_h, right_h):
    """Utility for steady Laplace tests using very small dt and S=0."""
    config = {
        "nx": nx,
        "ny": nx,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 1.0,
        "steps": 1,
        "T": 1.0,
        "S": 0.0,
        "initial_head": 0.0,
        "boundary_conditions": [
            {"type": "dirichlet", "cells": [(i,0) for i in range(nx)], "value": left_h},
            {"type": "dirichlet", "cells": [(i,nx-1) for i in range(nx)], "value": right_h},
        ],
    }
    heads = run_transient_solver(config)
    return heads[-1]
