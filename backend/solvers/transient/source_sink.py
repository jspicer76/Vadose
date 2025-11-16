# backend/solvers/transient/source_sink.py

from __future__ import annotations

from typing import List, Tuple, Any
import numpy as np

from .pumping_schedule import PumpingSchedule, make_schedule


class SourceSink:
    """
    Builds the right-hand-side (RHS) vector for the transient system,
    accounting for wells (Q(t)) and, later, time-varying recharge.

    Assumptions about the model:
      - model.grid has:
          * num_cells (int)
          * cell_index(row, col) -> flat index
      - model.wells is a list of well objects or dicts.
        Each well provides:
          * row/col or i/j (zero-based indices)
          * Q or q  (base pumping rate)
          * optional 'schedule' config:
              - attribute .schedule (object or dict), or
              - key 'schedule' when well is a dict.
    """

    def __init__(self, model: Any):
        self.model = model
        self.grid = model.grid
        self.n = self.grid.num_cells

        # List of (cell_index, schedule) pairs
        self._well_schedules: List[Tuple[int, PumpingSchedule]] = []

        self._build_well_schedules()
        # Recharge schedules could be added later in the same pattern

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_well_schedules(self) -> None:
        """Prepare (index, schedule) pairs for all wells in the model."""
        self._well_schedules.clear()

        wells = getattr(self.model, "wells", []) or []

        for w in wells:
            # --- row / col -------------------------------------------------
            row = None
            col = None

            # Object-style wells
            if hasattr(w, "row"):
                row = getattr(w, "row")
            elif hasattr(w, "i"):
                row = getattr(w, "i")

            if hasattr(w, "col"):
                col = getattr(w, "col")
            elif hasattr(w, "j"):
                col = getattr(w, "j")

            # Dict-style wells as fallback
            if (row is None or col is None) and isinstance(w, dict):
                row = w.get("row", w.get("i"))
                col = w.get("col", w.get("j"))

            # Skip if we still don't have valid indices
            if row is None or col is None:
                # You could log a warning here if desired
                continue

            # --- base Q ----------------------------------------------------
            base_q = None
            if hasattr(w, "Q"):
                base_q = getattr(w, "Q")
            elif hasattr(w, "q"):
                base_q = getattr(w, "q")
            elif isinstance(w, dict):
                base_q = w.get("Q", w.get("q", 0.0))

            if base_q is None:
                base_q = 0.0

            # --- schedule config -------------------------------------------
            schedule_cfg = None
            if hasattr(w, "schedule"):
                schedule_cfg = getattr(w, "schedule")
            elif isinstance(w, dict):
                schedule_cfg = w.get("schedule")

            # If schedule_cfg is already a PumpingSchedule, keep it
            if isinstance(schedule_cfg, PumpingSchedule.__constraints__ if hasattr(PumpingSchedule, "__constraints__") else ()):  # type: ignore
                schedule = schedule_cfg  # not typical branch; left for completeness
            else:
                # Expect config dict; factory will fall back to constant if None/invalid
                schedule = make_schedule(
                    config=schedule_cfg if isinstance(schedule_cfg, dict) else None,
                    default_q=float(base_q),
                )

            # --- map to global cell index ----------------------------------
            cell_idx = self.grid.cell_index(int(row), int(col))

            self._well_schedules.append((cell_idx, schedule))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def vector_at_time(self, t: float) -> np.ndarray:
        """
        Return the RHS vector f(t) from wells at time t.

        Sign convention:
          - Use your model's convention (typically negative for pumping).
        """
        rhs = np.zeros(self.n, dtype=float)

        # Wells
        for idx, schedule in self._well_schedules:
            rhs[idx] += schedule(t)

        # TODO (future): add time-varying recharge here.

        return rhs
