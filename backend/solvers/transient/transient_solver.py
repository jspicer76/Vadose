# backend/solvers/transient/transient_solver.py

import numpy as np

from .matrix_assembly import MatrixAssembly
from .storage import Storage
from .source_sink import SourceSink
from .solver_logs import SolverLogs
from .budget import BudgetEngine

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
                A_eff, b = bc.apply(A_eff, b)

            # Solve linear system
            h_new_vec = np.linalg.solve(A_eff, b)

            # Convert back to 2D grid
            h_new = h_new_vec.reshape((nx, ny))

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
