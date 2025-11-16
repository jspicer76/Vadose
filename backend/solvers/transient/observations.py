import numpy as np


class ObservationRecorder:
    """
    Collects head/drawdown time series at configured observation points.
    """

    def __init__(self, model):
        points = getattr(model, "observation_points", []) or []
        self.points = list(points)
        self.enabled = len(self.points) > 0
        self.times = []
        self._head = {pt.name: [] for pt in self.points}
        self._drawdown = {pt.name: [] for pt in self.points}

    def record(self, time_value, head_field):
        if not self.enabled:
            return

        self.times.append(float(time_value))
        for pt in self.points:
            h = float(head_field[pt.i, pt.j])
            self._head[pt.name].append(h)
            self._drawdown[pt.name].append(pt.reference_head - h)

    def results(self):
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

        for name in head:
            summary[name] = head[name]
            summary[f"{name}_drawdown"] = drawdown[name]

        return summary
