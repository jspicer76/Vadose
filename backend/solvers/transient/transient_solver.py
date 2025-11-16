# backend/solvers/transient/transient_solver.py

import numpy as np

from .matrix_assembly import MatrixAssembly
from .storage import Storage
from .source_sink import SourceSink
from .solver_logs import SolverLogs
from .budget import BudgetEngine
from .observations import ObservationRecorder
from .bc_factory import BCFactory
from .time_stepper import FixedTimeStepper, ImplicitEulerStepper

from backend.models.transient_model import TransientModel


class TransientSolver:
    """
    Transient groundwater flow solver.

    Implements:
        (S/dt + A) h^n = S/dt * h^(n-1) + f(t_n)

    where f(t) comes from wells and any other time-varying sources.
    """

    def __init__(self, model, time_stepper):
        if not isinstance(model, TransientModel):
            raise TypeError("TransientSolver: model must be a TransientModel")

        self.model = model
        self.time_stepper = time_stepper
        self.logs = SolverLogs()

        # Build components
        self.assembly = MatrixAssembly(model)
        self.storage = Storage(model)
        self.source_sink = SourceSink(model)
        self.budget = BudgetEngine(model)

    # ------------------------------------------------------------------
    def run(self, t_start: float, t_end: float, dt: float):
        """
        Advance solution in time. Returns:
            heads: list of head fields at each time step
            obs_data: observation point records
            logs: detailed solver logs
        """
        self.logs.start("transient")

        # Time vector
        times = np.arange(t_start, t_end + dt, dt)

        nsteps = len(times)
        n = self.model.grid.num_cells

        # Build static matrices
        A = self.assembly.matrix()     # conductance matrix
        S = self.storage.vector()      # storage coeffs per cell

        # Precompute S/dt
        S_over_dt = S / dt

        # LHS matrix: (S/dt + A)
        LHS = np.diag(S_over_dt) + A

        # Initial head
        h_prev = self.model.initial_head.copy()

        heads = [h_prev.copy()]

        # Observation recorder
        from .observations import ObservationRecorder
        obs_recorder = ObservationRecorder(self.model)

        for k in range(1, nsteps):
            t = times[k]

            # Right-hand side: S/dt * h_prev + f(t)
            rhs = S_over_dt * h_prev
            rhs += self.source_sink.vector_at_time(t)

            # Solve
            h_new = self.time_stepper.solve(LHS, rhs)

            # Store
            heads.append(h_new.copy())

            # Record observations
            obs_recorder.record(t, h_new)

            # Budget
            self.budget.record_step(k, t, h_new, h_prev, dt)

            # Logging
            self.logs.step(k, t, residual=None, note="OK")

            # Advance
            h_prev = h_new

        self.logs.finish("transient")

        # Package observation results
        obs_data = obs_recorder.get_results()

        return heads, obs_data, self.logs


def run_transient_solver(config):
    """
    Legacy helper used in regression tests.

    Args:
        config (dict): simulation parameters. Required keys:
            nx, ny, dx, dy, dt, T, S, initial_head
            steps or t_end must also be provided.

        Optional keys: pumping_wells, boundary_conditions, recharge,
        observation_points, t_start, return_full_output.

    Returns:
        np.ndarray or dict: heads array if return_full_output=False,
        otherwise a dict with heads/logs/observations.
    """
    required = ["nx", "ny", "dx", "dy", "dt", "T", "S", "initial_head"]
    for key in required:
        if key not in config:
            raise ValueError(f"run_transient_solver missing '{key}'")

    nx = config["nx"]
    ny = config["ny"]
    dx = config["dx"]
    dy = config["dy"]
    dt = float(config["dt"])
    t_start = float(config.get("t_start", 0.0))

    steps = config.get("steps")
    t_end = config.get("t_end")
    if steps is not None:
        t_end = t_start + steps * dt
    elif t_end is None:
        raise ValueError("Provide either 'steps' or 't_end' in config")
    else:
        t_end = float(t_end)

    model = TransientModel(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        T=config["T"],
        S=config["S"],
        h0=config["initial_head"],
        pumping_wells=config.get("pumping_wells"),
        boundary_conditions=[],
        recharge=config.get("recharge"),
        observation_points=config.get("observation_points"),
    )

    bc_configs = config.get("boundary_conditions", []) or []
    bc_objects = [BCFactory.create(bc_conf, model=model) for bc_conf in bc_configs]
    model.boundary_conditions = bc_objects

    stepper = FixedTimeStepper(t_start, t_end, dt)
    integrator = ImplicitEulerStepper()
    solver = TransientSolver(model, stepper, integrator)

    heads, obs_data, logs = solver.run(t_start, t_end, dt)

    if config.get("return_full_output", False):
        return {
            "heads": heads,
            "logs": logs,
            "observations": obs_data,
        }

    return heads
