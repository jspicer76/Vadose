from dataclasses import dataclass


@dataclass
class SolverParameters:
    """
    Global numerical / physical parameters for the solver.
    """
    dt: float = 1.0  # time step (T)
    max_iterations: int = 200
    tolerance: float = 1e-6
