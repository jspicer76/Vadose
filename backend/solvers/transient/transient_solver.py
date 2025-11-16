# backend/solvers/transient/transient_solver.py

import numpy as np

from .matrix_assembly import MatrixAssembly
from .storage import Storage
from .source_sink import SourceSink
from .solver_logs import SolverLogs
from .budget import BudgetEngine
from backend.models.transient_model import TransientModel


class TransientSolver:
    """
    Transient groundwater flow solver.
    Coordinates:
    - matrix assembly
    - storage coefficients
    - wells / recharge (source-sink)
    - boundary conditions
    - time stepping method
    """

    def __init__(self, model, time_stepper):
        self.model = model
        self.time_stepper = time_stepper
        self.logs = SolverLogs()

    def run(self, t_start, t_end, dt):
        """
        Run transient simulation from t_start to t_end with timestep dt.
        Returns:
            heads: array [ntimes, nx, ny]
            logs: SolverLogs object
        """

        nx, ny = self.model.nx, self.model.ny
        N = nx * ny

        budget_engine = BudgetEngine(self.model)


        # Initial condition
        h = self.model.h0.copy()
        heads = [h.copy()]

        # Time tracker
        t = t_start
        step = 0

        while t < t_end:
            # -------------------------
            # Build conductance matrix A
            # -------------------------
            A = MatrixAssembly.build_conductance_matrix(self.model)

            # -------------------------
            # Build W (source-sink array)
            # -------------------------
            W = SourceSink.build_W_vector(self.model, t)

            # -------------------------
            # Compute storage term for this step
            # -------------------------
            self.model.storage = Storage.compute_storage(self.model, h)

            # -------------------------
            # Construct linear system using time step method
            # A_eff h_new = b
            # -------------------------
            A_eff, b = self.time_stepper.build_system(
                self.model, h, dt, A, W
            )

            # Convert b to vector form
            b = b.flatten()

            # -------------------------
            # Apply boundary conditions (modify A_eff, b)
            # -------------------------
            for bc in self.model.boundary_conditions:
                if hasattr(bc, "update"):
                    bc.update(t + dt)
                A_eff, b = bc.apply(A_eff, b)

            # Solve linear system
            # Solve linear system
            h_new_vec = np.linalg.solve(A_eff, b)

            # Convert back to 2D grid
            h_new = h_new_vec.reshape((nx, ny))

            # Debug output for first step
            if step == 0:
                print("DEBUG: h_new range:", h_new.min(), h_new.max())


            # Mass balance summary printing every N steps
            if step % getattr(self.model, "budget_interval", 10) == 0:
                summary = budget_engine.summarize(step, t, dt, h, h_new)
                print(summary)

            # Update the head for the next step
            h = h_new

            # Save to history
            heads.append(h.copy())



            # -------------------------
            # Logging
            # -------------------------
            self.logs.log(t, dt, step, "Transient step completed")

            # Advance time
            t += dt
            step += 1

        return np.array(heads), self.logs
# -------------------------------------------------------------------
# Convenience API wrapper: run_transient_solver(config)
# -------------------------------------------------------------------

from backend.solvers.transient.time_stepper import ImplicitEulerStepper
from backend.models.transient_model import TransientModel   # you may need to adjust this import
from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC
from backend.solvers.transient.boundary_conditions.theis_dirichlet import TheisDirichletBC


def _perimeter_cells(nx: int, ny: int):
    """Return (linear indices, (i,j) coords) for outer perimeter in row-major storage."""
    cells = []
    coords = []
    # Top row (i = 0)
    for j in range(ny):
        cells.append(j)
        coords.append((0, j))
    # Bottom row (i = nx - 1)
    if nx > 1:
        for j in range(ny):
            idx = (nx - 1) * ny + j
            cells.append(idx)
            coords.append((nx - 1, j))
    # Left/right columns excluding corners already added
    for i in range(1, nx - 1):
        # Left column j = 0
        idx = i * ny
        cells.append(idx)
        coords.append((i, 0))
        # Right column j = ny - 1
        if ny > 1:
            idx = i * ny + (ny - 1)
            cells.append(idx)
            coords.append((i, ny - 1))
    return cells, coords


def run_transient_solver(config: dict):
    """
    High-level wrapper for running the transient groundwater solver.

    Parameters
    ----------
    config : dict
        Must define:
        - nx, ny
        - dx, dy
        - dt
        - steps
        - T, S
        - initial_head
        - pumping_wells (list)
        - boundary_conditions (optional)

    Returns
    -------
    heads : np.ndarray
        Array with shape [ntimes, nx, ny]
    """

    # ----------------------------------------------------------------
    # 1. Build model object
    # ----------------------------------------------------------------
    nx = config["nx"]
    ny = config["ny"]
    dx = config["dx"]
    dy = config["dy"]
    dt = config["dt"]
    steps = config["steps"]

    # Total time
    t_start = 0.0
    t_end = dt * steps

    # Must use your actual transient model class.
    # If the path/name is different, tell me and I’ll adjust it.
    boundary_conditions = config.get("boundary_conditions")
    if boundary_conditions is None:
        boundary_conditions = []

    # Optionally enforce constant-head boundary on outer perimeter to mimic infinite aquifer.
    if (
        not boundary_conditions
        and config.get("add_constant_head_boundary", True)
    ):
        perimeter_cells, perimeter_coords = _perimeter_cells(nx, ny)
        well_specs = [
            (w["i"], w["j"], w["Q"]) for w in config.get("pumping_wells", [])
        ]
        if config.get("use_theis_boundary", True) and well_specs:
            boundary_conditions = [
                TheisDirichletBC(
                    cells=perimeter_cells,
                    cell_coords=perimeter_coords,
                    wells=well_specs,
                    T=config["T"],
                    S=config["S"],
                    dx=dx,
                    dy=dy,
                    head0=config["initial_head"],
                )
            ]
        else:
            boundary_conditions = [
                DirichletBC(cells=perimeter_cells, head_value=config["initial_head"])
            ]

    model = TransientModel(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        T=config["T"],
        S=config["S"],
        h0=np.full((nx, ny), config["initial_head"]),
        pumping_wells=config.get("pumping_wells", []),
        boundary_conditions=boundary_conditions,
    )

    # ----------------------------------------------------------------
    # 2. Create time stepper
    # ----------------------------------------------------------------
    time_stepper = ImplicitEulerStepper()

    # ----------------------------------------------------------------
    # 3. Run transient solver
    # ----------------------------------------------------------------
    solver = TransientSolver(model, time_stepper)
    heads, logs = solver.run(t_start, t_end, dt)

    return heads
