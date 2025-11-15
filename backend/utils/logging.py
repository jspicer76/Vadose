"""
Centralized logging utilities for the Vadose groundwater backend.
Provides simple console logging used throughout backend modules.
"""

import datetime


# ---------------------------------------------------------
# Helper — timestamp
# ---------------------------------------------------------
def _ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


# ---------------------------------------------------------
# Solver Logging API (Required by M.1e)
# ---------------------------------------------------------
def log_solver_start(message: str):
    """Log when a solver begins."""
    print(f"{_ts()} [SOLVER START] {message}")


def log_solver_result(result: dict):
    """Log a summary of solver results."""
    converged = result.get("converged")
    iterations = result.get("iterations")
    residual = result.get("residual_norm")
    method = result.get("method")
    notes = result.get("notes", "")

    print(f"{_ts()} [SOLVER RESULT] Method: {method}")
    print(f"{_ts()}   Converged:      {converged}")
    print(f"{_ts()}   Iterations:     {iterations}")
    print(f"{_ts()}   Residual Norm:  {residual:.3e}")
    print(f"{_ts()}   Notes:          {notes}")


# ---------------------------------------------------------
# Optional general logging helpers
# ---------------------------------------------------------
def log_info(message: str):
    print(f"{_ts()} [INFO] {message}")


def log_warning(message: str):
    print(f"{_ts()} [WARNING] {message}")


def log_error(message: str):
    print(f"{_ts()} [ERROR] {message}")
