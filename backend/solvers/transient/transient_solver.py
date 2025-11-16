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

    Architecture:
      - stepper    → generates times (FixedTimeStepper)
      - integrator → builds A_eff, b (ImplicitEulerStepper)
    """

    def __init__(self, model, stepper, integrator):
        self.model = model
        self.stepper = stepper
        self.integrator = integrator
        self.logs = SolverLogs()
        self.observations = None

    # --------------------------------------------------------------
    def run(self, t_start, t_end, dt):
        """
        Execute transient simulation from t_start to t_end with time-step dt.
        Returns:
            heads: (ntime, nx, ny) array
            logs : SolverLogs instance
        """

        nx, ny = self.model.nx, self.model.ny
        budget_engine = BudgetEngine(self.model)

        # Initial head
        h = self.model.h0.copy()
        heads = [h.copy()]
        step = 0

        # Time sequence from the stepper
        times = self.stepper.times()
        obs_recorder = ObservationRecorder(self.model)
        obs_recorder.record(times[0], h)

        for idx in range(len(times) - 1):
            t  = times[idx]
            dt = times[idx + 1] - times[idx]

            # ----------------------------------------------------------
            # Conductance matrix A
            # ----------------------------------------------------------
            A = MatrixAssembly.build_conductance_matrix(self.model)

            # ----------------------------------------------------------
            # Volumetric source/sink W (wells + recharge)
            # ----------------------------------------------------------
            W = SourceSink.build_W_vector(self.model, t)

            # ----------------------------------------------------------
            # Storage term C = S*b*area
            # ----------------------------------------------------------
            self.model.storage = Storage.compute_storage(self.model, h)

            # ----------------------------------------------------------
            # Build the system A_eff h_new = b
            # ----------------------------------------------------------
            A_eff, b = self.integrator.build_system(
                self.model, h, dt, A, W
            )
            b = b.flatten()

            # ----------------------------------------------------------
            # Apply all boundary conditions
            # ----------------------------------------------------------
            for bc in self.model.boundary_conditions:

                # (1) time-dependent updates
                if hasattr(bc, "update") and callable(bc.update):
                    bc.update(t + dt)

                # (2) BCs with A/B modification (Dirichlet, GHB, River, Theis...)
                if hasattr(bc, "apply"):
                    A_eff, b = bc.apply(A_eff, b)

                # (3) Flux BC (Neumann): adds only to RHS
                if hasattr(bc, "apply_to_rhs"):
                    b = bc.apply_to_rhs(b, t)

            # ----------------------------------------------------------
            # Solve linear system
            # ----------------------------------------------------------
            h_new = np.linalg.solve(A_eff, b).reshape((nx, ny))

            # ----------------------------------------------------------
            # Debug + mass balance
            # ----------------------------------------------------------
            if step == 0:
                print("DEBUG: h_new range:", h_new.min(), h_new.max())

            if step % getattr(self.model, "budget_interval", 10) == 0:
                print(budget_engine.summarize(step, t, dt, h, h_new))

            # Prepare for next step
            h = h_new
            heads.append(h.copy())
            self.logs.log(t, dt, step, "Transient step completed")
            obs_recorder.record(times[idx + 1], h)
            step += 1

        obs_data = obs_recorder.results()
        self.logs.observations = obs_data
        self.observations = obs_data

        return np.array(heads), obs_data, self.logs


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
