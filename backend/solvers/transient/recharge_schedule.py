# backend/solvers/transient/recharge_schedule.py

from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional, Sequence
import math
import numpy as np


class RechargeSchedule(Protocol):
    """Callable interface for time-varying recharge R(t)."""

    def __call__(self, t: float) -> np.ndarray:
        ...


@dataclass
class ConstantRecharge:
    """Uniform constant recharge field."""
    R: float
    nx: int
    ny: int

    def __call__(self, t: float) -> np.ndarray:
        return np.full((self.nx, self.ny), self.R)


@dataclass
class StepRecharge:
    """Piecewise-constant R(t) schedule."""
    times: Sequence[float]
    rates: Sequence[float]
    nx: int
    ny: int

    def __call__(self, t: float) -> np.ndarray:
        if t <= self.times[0]:
            return np.full((self.nx, self.ny), self.rates[0])
        for i in range(len(self.times)-1, -1, -1):
            if t >= self.times[i]:
                return np.full((self.nx, self.ny), self.rates[i])
        return np.full((self.nx, self.ny), self.rates[0])


@dataclass
class SinusoidalRecharge:
    """Seasonal recharge pattern."""
    R_mean: float
    R_amp: float
    period: float
    phase: float
    nx: int
    ny: int

    def __call__(self, t: float) -> np.ndarray:
        arg = 2.0 * math.pi * (t - self.phase) / self.period
        R = self.R_mean + self.R_amp * math.sin(arg)
        return np.full((self.nx, self.ny), R)


def make_recharge_schedule(model, config: Optional[Dict[str, Any]]):
    """
    Builds a recharge callable from config dict.

    Examples:
        {"type": "constant", "R": 1e-8}
        {"type": "step", "times": [...], "rates": [...]}
        {"type": "sinusoidal", "R_mean":..., "R_amp":..., "period":...}
    """
    nx, ny = model.nx, model.ny

    if config is None:
        return None

    rtype = config.get("type", "constant").lower()

    if rtype == "constant":
        R = float(config.get("R", 0.0))
        return ConstantRecharge(R, nx, ny)

    if rtype == "step":
        return StepRecharge(
            times=list(map(float, config["times"])),
            rates=list(map(float, config["rates"])),
            nx=nx,
            ny=ny,
        )

    if rtype == "sinusoidal":
        return SinusoidalRecharge(
            R_mean=float(config["R_mean"]),
            R_amp=float(config["R_amp"]),
            period=float(config["period"]),
            phase=float(config.get("phase", 0.0)),
            nx=nx,
            ny=ny
        )

    return None
