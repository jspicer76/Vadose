# backend/solvers/transient/pumping_schedule.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Dict, Any, Optional
import math


class PumpingSchedule(Protocol):
    """
    Callable interface for time-varying pumping schedules.

    Usage:
        q = schedule(t)  # pumping rate at time t
    """

    def __call__(self, t: float) -> float:
        ...


@dataclass
class ConstantSchedule:
    """Q(t) = q for all t."""
    q: float

    def __call__(self, t: float) -> float:
        return float(self.q)


@dataclass
class StepSchedule:
    """
    Piecewise-constant schedule defined by change times and rates.

    times: sequence of times when the rate *changes*
    rates: sequence of rates, same length as times

    For t < times[0], we use rates[0].
    For times[k] <= t < times[k+1], we use rates[k].
    For t >= times[-1], we use rates[-1].
    """
    times: Sequence[float]
    rates: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.times) == 0:
            raise ValueError("StepSchedule: 'times' must not be empty")
        if len(self.times) != len(self.rates):
            raise ValueError("StepSchedule: 'times' and 'rates' must have same length")
        # Ensure times are sorted
        if any(self.times[i] > self.times[i + 1] for i in range(len(self.times) - 1)):
            raise ValueError("StepSchedule: 'times' must be non-decreasing")

    def __call__(self, t: float) -> float:
        # Before first time -> first rate
        if t <= self.times[0]:
            return float(self.rates[0])

        # Find last time <= t
        for i in range(len(self.times) - 1, -1, -1):
            if t >= self.times[i]:
                return float(self.rates[i])

        # Fallback (should never hit)
        return float(self.rates[0])


@dataclass
class RampSchedule:
    """
    Linear ramp from Q_start at t_start to Q_end at t_end.
    Outside [t_start, t_end], Q is clamped to endpoints.
    """
    t_start: float
    t_end: float
    Q_start: float
    Q_end: float

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError("RampSchedule: t_end must be > t_start")

    def __call__(self, t: float) -> float:
        if t <= self.t_start:
            return float(self.Q_start)
        if t >= self.t_end:
            return float(self.Q_end)

        # Linear interpolation
        frac = (t - self.t_start) / (self.t_end - self.t_start)
        return float(self.Q_start + frac * (self.Q_end - self.Q_start))


@dataclass
class SinusoidalSchedule:
    """
    Sinusoidal schedule:
        Q(t) = Q_mean + Q_amp * sin(2*pi*(t - phase) / period)

    period: time units (must be > 0)
    phase: time shift (same units as t)
    """
    Q_mean: float
    Q_amp: float
    period: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError("SinusoidalSchedule: period must be > 0")

    def __call__(self, t: float) -> float:
        arg = 2.0 * math.pi * (t - self.phase) / self.period
        return float(self.Q_mean + self.Q_amp * math.sin(arg))


@dataclass
class PulseSchedule:
    """
    Simple on/off pulsed schedule with fixed period.

    During each period:
        - Q = Q_on when t_in_period is in [t_on, t_off)
        - Q = Q_off otherwise

    period: length of one cycle (must be > 0)
    """
    Q_on: float
    Q_off: float
    period: float
    t_on: float
    t_off: float

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError("PulseSchedule: period must be > 0")
        if not (0.0 <= self.t_on < self.period):
            raise ValueError("PulseSchedule: 0 <= t_on < period required")
        if not (0.0 <= self.t_off <= self.period):
            raise ValueError("PulseSchedule: 0 <= t_off <= period required")
        if self.t_off <= self.t_on:
            raise ValueError("PulseSchedule: t_off must be > t_on")

    def __call__(self, t: float) -> float:
        # Map t into [0, period)
        t_mod = t % self.period
        if self.t_on <= t_mod < self.t_off:
            return float(self.Q_on)
        return float(self.Q_off)


def make_schedule(config: Optional[Dict[str, Any]], default_q: float) -> PumpingSchedule:
    """
    Factory to construct a PumpingSchedule from a config dict.

    If config is None, returns ConstantSchedule(default_q).

    Example configs:
        {"type": "constant", "Q": -500.0}

        {"type": "step",
         "times": [0.0, 3600.0, 7200.0],
         "rates": [-500.0, -1000.0, 0.0]}

        {"type": "ramp",
         "t_start": 0.0, "t_end": 86400.0,
         "Q_start": 0.0, "Q_end": -1500.0}

        {"type": "sinusoidal",
         "Q_mean": -800.0, "Q_amp": 200.0,
         "period": 86400.0, "phase": 0.0}

        {"type": "pulse",
         "Q_on": -1500.0, "Q_off": 0.0,
         "period": 14400.0,
         "t_on": 3600.0, "t_off": 7200.0}
    """
    if config is None:
        return ConstantSchedule(default_q)

    if not isinstance(config, dict):
        # Graceful fallback
        return ConstantSchedule(default_q)

    schedule_type = str(config.get("type", "constant")).lower()

    try:
        if schedule_type == "constant":
            q = float(config.get("Q", default_q))
            return ConstantSchedule(q)

        elif schedule_type == "step":
            times = config.get("times", [])
            rates = config.get("rates", [])
            return StepSchedule(times=list(map(float, times)),
                                rates=list(map(float, rates)))

        elif schedule_type == "ramp":
            return RampSchedule(
                t_start=float(config["t_start"]),
                t_end=float(config["t_end"]),
                Q_start=float(config.get("Q_start", default_q)),
                Q_end=float(config.get("Q_end", default_q)),
            )

        elif schedule_type == "sinusoidal":
            return SinusoidalSchedule(
                Q_mean=float(config.get("Q_mean", default_q)),
                Q_amp=float(config["Q_amp"]),
                period=float(config["period"]),
                phase=float(config.get("phase", 0.0)),
            )

        elif schedule_type == "pulse":
            return PulseSchedule(
                Q_on=float(config.get("Q_on", default_q)),
                Q_off=float(config.get("Q_off", 0.0)),
                period=float(config["period"]),
                t_on=float(config["t_on"]),
                t_off=float(config["t_off"]),
            )

    except (KeyError, TypeError, ValueError):
        # If anything goes wrong, fall back safely
        return ConstantSchedule(default_q)

    # Default fallback
    return ConstantSchedule(default_q)
