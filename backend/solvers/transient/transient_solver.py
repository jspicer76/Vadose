# backend/solvers/transient/transient_solver.py

import numpy as np
from numbers import Integral

from .matrix_assembly import MatrixAssembly
from .storage import Storage
from .source_sink import SourceSink
from .solver_logs import SolverLogs
from .budget import BudgetEngine

from backend.models.transient_model import TransientModel


class TransientSolver:
    """
    Transient groundwater flow solver.

    NEW ARCHITECTURE:
        - stepper generates times     (FixedTimeStepper)
        - integrator forms A_eff, b   (ImplicitEulerStepper)
    """

    def __init__(self, model, stepper, integrator):
        self.model = model
        self.stepper = stepper        # FixedTimeStepper
        self.integrator = integrator  # ImplicitEulerStepper / CN
        self.logs = SolverLogs()

    # --------------------------------------------------------------
    def run(self, t_start, t_end, dt):
        nx, ny = self.model.nx, self.model.ny
        budget_engine = BudgetEngine(self.model)

        h = self.model.h0.copy()
        heads = [h.copy()]
        step = 0

        # Get list of times from stepper
        times = self.stepper.times()

        for idx in range(len(times)-1):
            t = times[idx]
            dt = times[idx+1] - times[idx]

            # ----------------------------------------------
            # Conductance matrix
            # ----------------------------------------------
            A = MatrixAssembly.build_conductance_matrix(self.model)

            # ----------------------------------------------
            # Sources/sinks (flux per area)
            # ----------------------------------------------
            W = SourceSink.build_W_vector(self.model, t)

            # ----------------------------------------------
            # Storage term
            # ----------------------------------------------
            self.model.storage = Storage.compute_storage(self.model, h)

            # ----------------------------------------------
            # Build system using integrator
            # ----------------------------------------------
            A_eff, b = self.integrator.build_system(
                self.model, h, dt, A, W
            )
            b = b.flatten()

            # ----------------------------------------------
            # Boundary conditions
            # ----------------------------------------------
            for bc in self.model.boundary_conditions:
                if hasattr(bc, "update"):
                    bc.update(t + dt)
                A_eff, b = bc.apply(A_eff, b)

            # ----------------------------------------------
            # Solve A_eff h_new = b
            # ----------------------------------------------
            h_new = np.linalg.solve(A_eff, b).reshape((nx, ny))

            # ----------------------------------------------
            # Debug / mass balance
            # ----------------------------------------------
            if step == 0:
                print("DEBUG: h_new range:", h_new.min(), h_new.max())

            if step % getattr(self.model, "budget_interval", 10) == 0:
                print(budget_engine.summarize(step, t, dt, h, h_new))

            h = h_new
            heads.append(h.copy())
            self.logs.log(t, dt, step, "Transient step completed")
            step += 1

        return np.array(heads), self.logs
