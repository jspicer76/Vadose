# backend/solvers/transient/time_stepper.py

from abc import ABC, abstractmethod
import numpy as np


class TimeStepper(ABC):
    """
    Abstract base class for transient time integration methods.
    """

    @abstractmethod
    def build_system(self, model, h_old, dt, A, W):
        """
        Construct A_eff and b for the linear system:
            A_eff h_new = b

        Must return:
            (A_eff, b)
        """
        pass


class ImplicitEulerStepper(TimeStepper):
    """
    Correct implicit (backward Euler) scheme for confined/unconfined flow.
    """

    def build_system(self, model, h_old, dt, A, W):
        S = model.storage                # array (nx, ny)
        storage_diag = S.flatten() / dt

        # Effective matrix
        A_eff = A + np.diag(storage_diag)

        # RHS vector
        b = storage_diag * h_old.flatten() + W.flatten()

        return A_eff, b


class CrankNicolsonStepper(TimeStepper):
    """
    Optional second-order time integrator.
    """

    def build_system(self, model, h_old, dt, A, W):
        S = model.storage
        S_diag = S.flatten() / dt

        # CN: (S/dt)*h^{n+1} - 0.5 A h^{n+1} = (S/dt)*h^n + 0.5 A h^n + W
        A_eff = np.diag(S_diag) - 0.5 * A

        b = (
            S_diag * h_old.flatten()
            + 0.5 * A @ h_old.flatten()
            + W.flatten()
        )

        return A_eff, b
