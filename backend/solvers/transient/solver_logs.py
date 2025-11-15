# backend/solvers/transient/solver_logs.py

class SolverLogs:
    """
    Handles logging of each timestep iteration.
    """

    def __init__(self):
        self.entries = []

    def log(self, t, dt, iteration, notes):
        entry = {
            "time": t,
            "dt": dt,
            "iteration": iteration,
            "notes": notes
        }
        self.entries.append(entry)

    def print_last(self):
        if self.entries:
            print(self.entries[-1])
