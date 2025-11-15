"""
Smoke Test for Vadose Groundwater Backend (Module M.1e)
"""

import numpy as np

from backend.core.grid import Grid
from backend.core.aquifer_model import AquiferModel
from backend.core.properties import AquiferProperties


def build_simple_model():
    """Construct a simple uniform 3×3 model."""

    # ------------------------------------------------
    # Build grid correctly for your Grid class
    # ------------------------------------------------
    dx = np.array([100.0, 100.0, 100.0])   # 3 columns
    dy = np.array([100.0, 100.0, 100.0])   # 3 rows

    grid = Grid(dx=dx, dy=dy, nlay=1)

    # ------------------------------------------------
    # Build AquiferProperties (1 layer, 3x3 cells)
    # ------------------------------------------------
    K = 100.0
    shape = (1, grid.nx, grid.ny)

    Kx = np.ones(shape) * K
    Ky = np.ones(shape) * K
    Kz = np.ones(shape) * K

    thickness = np.array([50.0])
    Sy = np.array([0.20])
    Ss = np.array([1e-5])

    props = AquiferProperties(
        Kx=Kx, Ky=Ky, Kz=Kz,
        thickness=thickness,
        Sy=Sy, Ss=Ss,
        confined=False
    )

    # ------------------------------------------------
    # Boundary conditions (simple fixed-head box)
    # ------------------------------------------------
    boundaries = [
        {"type": "CONSTANT_HEAD", "value": 100.0, "location": "LEFT"},
        {"type": "CONSTANT_HEAD", "value": 100.0, "location": "RIGHT"},
        {"type": "CONSTANT_HEAD", "value": 100.0, "location": "TOP"},
        {"type": "CONSTANT_HEAD", "value": 100.0, "location": "BOTTOM"},
    ]

    return AquiferModel(
        name="Test Aquifer",
        grid=grid,
        properties=props,
        boundaries=boundaries,
        wells=[]
    )


def run_and_print(model, method):
    print(f"\n=== Running {method.upper()} solver ===")
    result = model.solve_steady_state(method=method)

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Residual Norm:", result["residual_norm"])
    print("Head Field:\n", result["head"])


if __name__ == "__main__":
    print("=== VADOSE SMOKE TEST ===")

    model = build_simple_model()
    run_and_print(model, "direct")
    run_and_print(model, "sor")

    print("\nSmoke test complete.\n")
