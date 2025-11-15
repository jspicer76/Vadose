import numpy as np

from backend.core.grid import Grid
from backend.core.properties import AquiferProperties
from backend.core.well import Well
from backend.solvers.steady_state.interface import solve_steady_state


def build_test_model():
    """
    Construct a simple 3x3 aquifer model for solver testing:
    - confined aquifer
    - uniform K
    - one constant-head boundary on left side
    - one pumping well in center
    """

    # -----------------------------
    # 1. Grid (Option B: flexible grid spacing)
    # -----------------------------
    dx = np.array([10.0, 10.0, 10.0])
    dy = np.array([10.0, 10.0, 10.0])
    grid = Grid(dx=dx, dy=dy, nlay=1)

    # -----------------------------
    # 2. Hydraulic properties
    # -----------------------------
    nx, ny = grid.nx, grid.ny
    Kx = np.ones((1, nx, ny)) * 10.0  # m/day
    Ky = np.ones((1, nx, ny)) * 10.0
    Kz = np.ones((1, nx, ny)) * 1.0

    thickness = np.array([20.0])  # meters

    props = AquiferProperties(
        Kx=Kx,
        Ky=Ky,
        Kz=Kz,
        thickness=thickness,
        Sy=np.array([0.20]),
        Ss=np.array([1e-5]),
        porosity=np.array([0.25]),
        confined=True  # confined for this test
    )

    # -----------------------------
    # 3. Build AquiferModel object
    # -----------------------------
    from backend.core.aquifer_model import AquiferModel

    model = AquiferModel(
        name="Test Aquifer M1d",
        grid=grid,
        properties=props,
        wells=[],
        boundaries=[]
    )

    # -----------------------------
    # 4. Add constant-head boundaries (left side)
    # -----------------------------
    for j in range(ny):
        model.boundaries.append({
            "type": "CONSTANT_HEAD",
            "i": 0,
            "j": j,
            "value": 100.0
        })

    # -----------------------------
    # 5. Add a pumping well in the center
    # -----------------------------
    # Well at center grid location (i=1, j=1)
    well = Well(
        name="PW-1",
        x=grid.x_centers[1],
        y=grid.y_centers[1],
        rate=-500.0  # pumping (negative)
    )
    well.assign_to_grid(grid)
    model.wells.append(well)

    return model


def run_test():
    model = build_test_model()
    grid = model.grid

    print("\n=== STEADY-STATE GROUNDWATER SOLVER TEST ===")

    # --------------------------------------------------------
    # DIRECT SOLVER
    # --------------------------------------------------------
    h_direct, outer1, info1, ok1 = solve_steady_state(
        model,
        method="direct",
        verbose=False
    )

    print("\n[Direct Solver] Converged:", ok1)
    print("[Direct Solver] Outer iterations:", outer1)
    print("[Direct Solver] Head field:\n", h_direct)

    # --------------------------------------------------------
    # SOR ITERATIVE SOLVER
    # --------------------------------------------------------
    h_sor, outer2, info2, ok2 = solve_steady_state(
        model,
        method="sor",
        w=1.4,
        verbose=False
    )

    print("\n[SOR Solver] Converged:", ok2)
    print("[SOR Solver] Outer iterations:", outer2)
    print("[SOR Solver] SOR inner iterations:", info2["iters"])
    print("[SOR Solver] Head field:\n", h_sor)

    # --------------------------------------------------------
    # CHECK SOLUTION AGREEMENT
    # --------------------------------------------------------
    diff = np.max(np.abs(h_direct - h_sor))
    print("\nMax difference between direct & SOR solutions:", diff)

    if diff < 1e-3:
        print("PASS: Solutions match within tolerance.")
    else:
        print("WARNING: Solutions differ. Investigate further.")


if __name__ == "__main__":
    run_test()
