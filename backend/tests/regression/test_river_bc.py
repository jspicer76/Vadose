import numpy as np
from backend.solvers.transient.transient_solver import run_transient_solver

def test_ghb_flux_direction():
    nx = 3
    C = 1e-3
    hb = 10.0
    h0 = np.full((nx,nx), 12.0)

    config = {
        "nx": nx,
        "ny": nx,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 1.0,
        "steps": 1,
        "T": 1.0,
        "S": 0.0,
        "initial_head": h0,
        "boundary_conditions": [
            {"type": "ghb", "cells": [(1,1)], "stage": hb, "C": C},
        ],
    }

    # Run solver
    run_transient_solver(config)

    # Expected flux
    expected = C * (hb - 12.0)
    assert expected < 0   # outflow
