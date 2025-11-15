# backend/solvers/transient/time_stepper.py

from abc import ABC, abstractmethod
import numpy as np

class TimeStepper(ABC):
    """
    Base class for time integration methods.
    """

    @abstractmethod
    def build_system(self, model, h_old, dt, A, W):
        """
        Constructs A_eff and b for the linear system:
            A_eff h_new = b
        Returns: (A_eff, b)
        """
        pass


class ImplicitEulerStepper(TimeStepper):
    """
    Default implicit (backward Euler) integrator.
    Unconditionally stable.
    """

    def build_system(self, model, h_old, dt, A, W):
        S = model.storage  # array of S_ij values

        # A_eff = A + S/dt * I
        I = np.eye(A.shape[0])
        A_eff = A + np.diag(S.flatten() / dt)

        # b = S/dt * h_old + W
        b = (S.flatten() / dt) * h_old.flatten() + W.flatten()

        return A_eff, b


class CrankNicolsonStepper(TimeStepper):
    """
    Optional second-order accurate method.
    """

    def build_system(self, model, h_old, dt, A, W):
        S = model.storage
        I = np.eye(A.shape[0])

        A_eff = (np.diag(S.flatten()/dt) - 0.5 * A)
        b = (np.diag(S.flatten()/dt) + 0.5 * A) @ h_old.flatten() + W.flatten()

        return A_eff, b
