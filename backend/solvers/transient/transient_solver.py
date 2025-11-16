# backend/solvers/transient/transient_solver.py

import numpy as np
from numbers import Integral

from .matrix_assembly import MatrixAssembly
from .storage import Storage
from .source_sink import SourceSink
from .solver_logs import SolverLogs
from .budget import BudgetEngine

from backend.models.transient_model import TransientModel
from backend.solvers.transient.time_stepper import ImplicitEulerStepper


# ======================================================================
#                         TRANSIENT SOLVER CLASS
# ======================================================================

class TransientSolver:
    """
    Transient groundwater flow solver.
    Handles:
        - Matrix assembly
        - Storage
        - Wells & recharge
        - Boundary conditions
        - Time stepping
    """

    def __init__(self, model, time_stepper):
        self.model = model
        self.time_stepper = time_stepper
        self.logs = SolverLogs()

    # ------------------------------------------------------------------
    def run(self, t_start, t_end, dt):
        """
        Loop through transient steps.
        Returns heads[t][i][j] and logs.
        """

        nx, ny = self.model.nx, self.model.ny
        budget_engine = BudgetEngine(self.model)

        h = self.model.h0.copy()
        heads = [h.copy()]

        t = t_start
        step = 0

        while t < t_end:

            # ----------------------------------------------------------
            # Conductance matrix
            # ----------------------------------------------------------
            A = MatrixAssembly.build_conductance_matrix(self.model)

            # ----------------------------------------------------------
            # Build W (sources/sinks)
            # ----------------------------------------------------------
            W = SourceSink.build_W_vector(self.model, t)

            # ----------------------------------------------------------
            # Storage term
            # ----------------------------------------------------------
            self.model.storage = Storage.compute_storage(self.model, h)

            # ----------------------------------------------------------
            # Build system A_eff h_new = b
            # ----------------------------------------------------------
            A_eff, b = self.time_stepper.build_system(
                self.model, h, dt, A, W
            )

            b = b.flatten()

            # ----------------------------------------------------------
            # Apply boundary conditions
            # ----------------------------------------------------------
            for bc in self.model.boundary_conditions:
                # If BC has update(time): use it
                update_fn = getattr(bc, "update", None)
                if callable(update_fn):
                    update_fn(t + dt)

                # Apply BC to matrix system
                A_eff, b = bc.apply(A_eff, b)

            # ----------------------------------------------------------
            # Solve linear system
            # ----------------------------------------------------------
            h_new_vec = np.linalg.solve(A_eff, b)
            h_new = h_new_vec.reshape((nx, ny))

            # Debug for first step
            if step == 0:
                print("DEBUG: h_new range:", h_new.min(), h_new.max())

            # Mass balance
            if step % getattr(self.model, "budget_interval", 10) == 0:
                print(budget_engine.summarize(step, t, dt, h, h_new))

            # Prepare next step
            h = h_new
            heads.append(h.copy())

            # Logs
            self.logs.log(t, dt, step, "Transient step completed")

            t += dt
            step += 1

        return np.array(heads), self.logs


# ======================================================================
#                     HELPER: PERIMETER INDEX BUILDER
# ======================================================================

def _perimeter_cells(nx: int, ny: int):
    """Return (linear indices, (i,j) coords) for the perimeter."""
    cells = []
    coords = []

    # Top row
    for j in range(ny):
        cells.append(j)
        coords.append((0, j))

    # Bottom row
    if nx > 1:
        for j in range(ny):
            idx = (nx - 1) * ny + j
            cells.append(idx)
            coords.append((nx - 1, j))

    # Vertical edges
    for i in range(1, nx - 1):
        # Left edge
        idx = i * ny
        cells.append(idx)
        coords.append((i, 0))
        # Right edge
        idx = i * ny + (ny - 1)
        cells.append(idx)
        coords.append((i, ny - 1))

    return cells, coords


# ======================================================================
#               HIGH-LEVEL API: run_transient_solver(config)
# ======================================================================

from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC
from backend.solvers.transient.boundary_conditions.theis_dirichlet import TheisDirichletBC
from backend.solvers.transient.bc_factory import BCFactory


def _cell_to_linear_index(cell, ny):
    """
    Convert either a flat index or an (i, j) tuple into the solver's
    flattened matrix index.
    """
    if isinstance(cell, Integral):
        return int(cell)

    # NumPy scalars behave like Integrals but might not be caught above
    if hasattr(cell, "item") and np.isscalar(cell):
        return int(cell)

    if isinstance(cell, np.ndarray):
        cell = cell.tolist()

    if isinstance(cell, (tuple, list)):
        if len(cell) != 2:
            raise ValueError(f"Cell coordinates must be length 2: {cell}")
        i, j = cell
        return int(i) * ny + int(j)

    raise TypeError(f"Unsupported cell specifier: {cell!r}")


def _normalize_cells(cells, ny):
    """Normalize a list/iterable of cells to flattened indices."""
    return [_cell_to_linear_index(c, ny) for c in cells]


def run_transient_solver(config: dict):
    """
    High-level wrapper that:
        - Builds the model
        - Converts BC configs → BC objects
        - Runs the transient solver
    """

    nx = config["nx"]
    ny = config["ny"]
    dx = config["dx"]
    dy = config["dy"]
    dt = config["dt"]
    steps = config["steps"]

    t_start = 0.0
    t_end = dt * steps

    # ----------------------------------------------------------
    # Build boundary condition objects
    # ----------------------------------------------------------
    raw_bcs = config.get("boundary_conditions", [])
    bcs = []

    for bc_conf in raw_bcs:
        if isinstance(bc_conf, dict):
            bc_copy = bc_conf.copy()
            if "cells" in bc_copy:
                bc_copy["cells"] = _normalize_cells(bc_copy["cells"], ny)
            bcs.append(BCFactory.create(bc_copy))
        else:
            bcs.append(bc_conf)

    # Optional perimeter BCs (default on for pumping-well configs only)
    add_const = config.get("add_constant_head_boundary")
    if add_const is None:
        add_const = bool(config.get("pumping_wells"))

    if not bcs and add_const:
        per_cells, per_coords = _perimeter_cells(nx, ny)

        well_specs = [(w["i"], w["j"], w["Q"])
                      for w in config.get("pumping_wells", [])]

        if config.get("use_theis_boundary", True) and well_specs:
            bcs = [
                TheisDirichletBC(
                    cells=per_cells,
                    cell_coords=per_coords,
                    wells=well_specs,
                    T=config["T"],
                    S=config["S"],
                    dx=dx,
                    dy=dy,
                    head0=config["initial_head"],
                )
            ]
        else:
            bcs = [
                DirichletBC(
                    cells=per_cells,
                    head_value=config["initial_head"]
                )
            ]

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    model = TransientModel(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        T=config["T"],
        S=config["S"],
        h0=np.array(config["initial_head"], dtype=float),
        pumping_wells=config.get("pumping_wells", []),
        boundary_conditions=bcs,
    )

    # Recharge array
    if "recharge" in config:
        model.recharge = config["recharge"]

    # ----------------------------------------------------------
    # Run solver
    # ----------------------------------------------------------
    solver = TransientSolver(model, ImplicitEulerStepper())
    heads, logs = solver.run(t_start, t_end, dt)
    return heads
