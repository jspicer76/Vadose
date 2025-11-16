# backend/solvers/transient/solver_logs.py

import time

class SolverLogs:
    """
    Unified logging interface for steady-state and transient solvers.

    Supports:
      - start(tag)
      - step(k, t, residual, note)
      - finish(tag)
      - legacy log(t, dt, iteration, notes)
    """

    def __init__(self):
        # Holds all log entries (step-by-step)
        self.entries = []

        # Optional observation data
        self.observations = None

        # Start/end timestamps
        self.start_time = None
        self.end_time = None

        # Solver type tag ("transient", "steady", etc.)
        self.tag = None

    # ------------------------------------------------------------------
    # New API expected by TransientSolver
    # ------------------------------------------------------------------
    def start(self, tag: str):
        """Mark the beginning of a solver run."""
        self.tag = tag
        self.start_time = time.time()
        self.entries.append({
            "event": "start",
            "tag": tag,
            "timestamp": self.start_time
        })

    def step(self, step_index: int, t: float, residual, note=""):
        """Log a single transient time-step iteration."""
        self.entries.append({
            "event": "step",
            "step": step_index,
            "time": float(t),
            "residual": residual,
            "note": note
        })

    def finish(self, tag: str):
        """Mark the end of a solver run."""
        self.end_time = time.time()
        self.entries.append({
            "event": "finish",
            "tag": tag,
            "timestamp": self.end_time,
            "duration_sec": self.end_time - self.start_time
        })

    # ------------------------------------------------------------------
    # Backward Compatibility (existing tests may rely on this)
    # ------------------------------------------------------------------
    def log(self, t, dt, iteration, notes):
        """Legacy log method maintained for compatibility."""
        entry = {
            "event": "legacy",
            "time": t,
            "dt": dt,
            "iteration": iteration,
            "note": notes
        }
        self.entries.append(entry)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def print_last(self):
        if self.entries:
            print(self.entries[-1])

    def to_dict(self):
        """Return logs as a JSON-serializable structure."""
        return {
            "tag": self.tag,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "entries": self.entries
        }
