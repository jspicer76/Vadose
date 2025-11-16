# smoke_test_pumping_schedules.py

import numpy as np

from backend.solvers.transient.source_sink import SourceSink
from backend.solvers.transient.pumping_schedule import make_schedule


class FakeGrid:
    def __init__(self):
        self.num_cells = 4  # 2x2 grid, just for demo
        self.nrow = 2
        self.ncol = 2

    def cell_index(self, row: int, col: int) -> int:
        # row-major flattening
        return row * self.ncol + col


class FakeModel:
    def __init__(self):
        self.grid = FakeGrid()

        # Define a couple of wells with different schedules
        self.wells = [
            {
                "name": "StepTestWell",
                "row": 0,
                "col": 0,
                "Q": -500.0,
                "schedule": {
                    "type": "step",
                    "times": [0.0, 3600.0, 7200.0],
                    "rates": [-500.0, -1000.0, 0.0],
                },
            },
            {
                "name": "SinusoidWell",
                "row": 1,
                "col": 1,
                "Q": -800.0,
                "schedule": {
                    "type": "sinusoidal",
                    "Q_mean": -800.0,
                    "Q_amp": 200.0,
                    "period": 86400.0,
                    "phase": 0.0,
                },
            },
        ]


def main():
    model = FakeModel()
    src = SourceSink(model)

    times = [0.0, 1800.0, 3600.0, 5400.0, 7200.0, 86400.0]

    print("=== PUMPING SCHEDULE SMOKE TEST ===")
    for t in times:
        rhs = src.vector_at_time(t)
        print(f"t = {t:8.1f} s -> RHS = {rhs}")

    print("\nDone.")


if __name__ == "__main__":
    main()
