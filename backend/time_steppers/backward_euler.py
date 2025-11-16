import numpy as np

class BackwardEuler:
    """
    Simple linear Backward Euler solver:
        (S/dt + A) h^n = S/dt * h^(n-1) + f(t)
    """

    def solve(self, LHS: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(LHS, rhs)
