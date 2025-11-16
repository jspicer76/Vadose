# backend/solvers/transient/observations.py

import numpy as np


class ObservationRecorder:
    """
    Collects head/drawdown time series at configured observation points.
    """

    def __init__(self, model):
        # Model may define observation_points as:
        #  - list of ObservationPoint objects
        #  - list of dicts {"i":..., "j":..., "name":..., "reference_head":...}
        points = getattr(model, "observation_points", []) or []

        # Normalize points into a uniform structure
        self.points = []
        for pt in points:
            if hasattr(pt, "i") and hasattr(pt, "j"):  
                # ObservationPoint object
                self.points.append(pt)
            elif isinstance(pt, dict):
                # Convert dict to simple struct-like object
                Simple = type("SimpleObs", (), {})
                obj = Simple()
                obj.i = int(pt["i"])
                obj.j = int(pt["j"])
                obj.name = pt.get("name", f"obs_{len(self.points)+1}")
                obj.reference_head = float(pt.get("reference_head", 0.0))
                obj.cell = (obj.i, obj.j)
                self.points.append(obj)
            else:
                raise ValueError(f"Unsupported observation point type: {pt}")

        self.enabled = len(self.points) > 0
        self.times = []

        self._head = {pt.name: [] for pt in self.points}
        self._drawdown = {pt.name: [] for pt in self.points}

    # --------------------------------------------------------------
    # RECORD A TIME STEP
    # --------------------------------------------------------------
    def record(self, time_value, head_field):
        """
        Record head values at observation points for time t.
        head_field must be a 2D array (nx, ny).
        """
        if not self.enabled:
            return

        self.times.append(float(time_value))

        for pt in self.points:
            h = float(head_field[pt.i, pt.j])
            self._head[pt.name].append(h)
            self._drawdown[pt.name].append(pt.reference_head - h)

    # --------------------------------------------------------------
    # GET RESULTS (preferred interface)
    # --------------------------------------------------------------
    def get_results(self):
        """Wrapper for results() for compatibility with the solver."""
        return self.results()

    # --------------------------------------------------------------
    # BUILD STRUCTURED RESULTS
    # --------------------------------------------------------------
    def results(self):
        """
        Return dictionary summary including:
            - times
            - head time series
            - drawdown time series
            - metadata for each point
        """
        if not self.enabled:
            empty = np.array([], dtype=float)
            return {
                "_times": empty,
                "head": {},
                "drawdown": {},
                "points": [],
            }

        times = np.asarray(self.times, dtype=float)

        head = {
            name: np.asarray(values, dtype=float)
            for name, values in self._head.items()
        }
        drawdown = {
            name: np.asarray(values, dtype=float)
            for name, values in self._drawdown.items()
        }

        summary = {
            "_times": times,
            "head": head,
            "drawdown": drawdown,
            "points": [
                {
                    "name": pt.name,
                    "cell": pt.cell,
                    "reference_head": pt.reference_head,
                }
                for pt in self.points
            ],
        }

        # Flatten into top-level keys for convenience
        for name in head:
            summary[name] = head[name]
            summary[f"{name}_drawdown"] = drawdown[name]

        return summary
