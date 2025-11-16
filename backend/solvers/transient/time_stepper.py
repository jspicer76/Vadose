# backend/solvers/transient/time_stepper.py

import numpy as np
from abc import ABC, abstractmethod


# ================================================================
# Abstract Base TimeStepper
# ================================================================
class TimeStepper(ABC):
    """
    Abstract base class for transient time integration methods.

    Expected call signature from TransientSolver:
        build_system(model, h_old, dt, A, W)

    Must return:
        A_eff, b
    """

    @abstractmethod
    def build_system(self, model, h_old, dt, A, W):
        pass


# ================================================================
# Implicit Backward Euler (stable, 1st order)
# ================================================================
class ImplicitEulerStepper(TimeStepper):
    """
    Backward Euler:
        (C + dt*A) h_new = C*h_old + dt*W
    where:
        C = storage coefficient (S*b*area)
        A = conductance matrix
        W = volumetric sources/sinks
    """

    def build_system(self, model, h_old, dt, A, W):
        # S*b*area is already computed in model.storage
        # so DO NOT divide by dt and DO NOT multiply by area again
        C = model.storage.flatten()               # already volumetric storage coefficient

        # Left-hand side:
        #   A_eff = dt*A + C*I
        A_eff = dt * A + np.diag(C)

        # Right-hand side:
        #   b = C*h_old + dt*W
        b = C * h_old.flatten() + dt * W.flatten()

        return A_eff, b



# ================================================================
# Crank–Nicolson (2nd order)
# ================================================================
class CrankNicolsonStepper(TimeStepper):
    """
    Crank–Nicolson:
        (S/dt - 0.5 A) h_new = (S/dt) h_old + 0.5 A h_old + W
    """

    def build_system(self, model, h_old, dt, A, W):
        S = model.storage
        S_diag = S.flatten() / dt

        A_eff = np.diag(S_diag) - 0.5 * A

        b = (
            S_diag * h_old.flatten()
            + 0.5 * A @ h_old.flatten()
            + W.flatten()
        )

        return A_eff, b


# ================================================================
# Fixed Time Step Iterator
# ================================================================
class FixedTimeStepper:
    """
    Produces time list: t_start, t_start+dt, ..., t_end.

    This class is NOT the integrator.
    It simply generates time stamps.
    """

    def __init__(self, t_start, t_end, dt):
        self.t_start = float(t_start)
        self.t_end   = float(t_end)
        self.dt      = float(dt)

    def times(self):
        t = self.t_start
        out = [t]
        while t < self.t_end:
            t += self.dt
            out.append(t)
        return out
