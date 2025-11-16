# backend/solvers/transient/source_sink.py

from __future__ import annotations

from typing import List, Tuple, Any
import numpy as np

from .pumping_schedule import PumpingSchedule, make_schedule
from .recharge_schedule import make_recharge_schedule, ConstantRecharge


class SourceSink:
    """
    Builds the right-hand-side (RHS) vector for the transient system,
    accounting for:
        - Wells Q(t)
        - Recharge R(t)
    """

    def __init__(self, model: Any):
        self.model = model
        self.grid = model
        self.n = model.nx * model.ny

        # --------------------------------------------------------------
        # Build pumping schedules
        # --------------------------------------------------------------
        self._well_schedules: List[Tuple[int, PumpingSchedule]] = []
        self._build_well_schedules()

        # --------------------------------------------------------------
        # Build recharge schedule (R(t))
        # --------------------------------------------------------------
        self.recharge_schedule = None
        R = getattr(model, "recharge", None)

        if R is None:
            self.recharge_schedule = None

        elif isinstance(R, dict):
            # Time-varying recharge schedule
            self.recharge_schedule = make_recharge_schedule(model, R)

        elif np.isscalar(R):
            # Constant scalar recharge (uniform)
            self.recharge_schedule = ConstantRecharge(R, model.nx, model.ny)

        elif isinstance(R, np.ndarray):
            # Constant spatial recharge field
            self.recharge_schedule = lambda t, R=R: R

        else:
            raise TypeError(f"Unsupported recharge format: {type(R)}")

    # ------------------------------------------------------------------
    # INTERNAL — build well schedule list
    # ------------------------------------------------------------------
    def _build_well_schedules(self) -> None:
        """Prepare (cell_idx, schedule_fn) entries for wells."""
        self._well_schedules.clear()
        wells = getattr(self.model, "wells", []) or []

        for w in wells:
            # Extract well location (row,col)
            row = getattr(w, "row", getattr(w, "i", None))
            col = getattr(w, "col", getattr(w, "j", None))

            if isinstance(w, dict):
                row = w.get("row", w.get("i", row))
                col = w.get("col", w.get("j", col))

            if row is None or col is None:
                continue

            # Extract pumping rate
            if hasattr(w, "Q"):
                base_q = float(w.Q)
            elif hasattr(w, "q"):
                base_q = float(w.q)
            elif isinstance(w, dict):
                base_q = float(w.get("Q", w.get("q", 0.0)))
            else:
                base_q = 0.0

            # Extract schedule config
            if hasattr(w, "schedule"):
                schedule_cfg = w.schedule
            elif isinstance(w, dict):
                schedule_cfg = w.get("schedule")
            else:
                schedule_cfg = None

            # Build schedule
            schedule = make_schedule(
                config=schedule_cfg if isinstance(schedule_cfg, dict) else None,
                default_q=base_q,
            )

            # Convert cell (i,j) → global index
            idx = self.grid.cell_index(int(row), int(col))
            self._well_schedules.append((idx, schedule))

    # ------------------------------------------------------------------
    # PUBLIC — assemble f(t)
    # ------------------------------------------------------------------
    def vector_at_time(self, t: float) -> np.ndarray:
        """
        Build RHS vector f(t) from:
            - Wells Q(t)
            - Recharge R(t)
        """
        rhs = np.zeros(self.n, dtype=float)

        # --------------------------------------------------------------
        # Wells Q(t)
        # --------------------------------------------------------------
        for idx, schedule in self._well_schedules:
            rhs[idx] += schedule(t)

        # --------------------------------------------------------------
        # Recharge R(t)
        # --------------------------------------------------------------
        if self.recharge_schedule is not None:
            R_field = self.recharge_schedule(t)  # shape (nx, ny)
            cell_area = self.grid.dx * self.grid.dy
            rhs += R_field.flatten() * cell_area

        return rhs
