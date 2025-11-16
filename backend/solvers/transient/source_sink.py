# backend/solvers/transient/source_sink.py

import numpy as np

class SourceSink:
    """
    Handles wells, injections, recharge-as-sources, etc.
    """

    @staticmethod
    def build_W_vector(model, t):
        """
        Returns W array same shape as model.h of volumetric flux PER UNIT AREA.
        Positive = injection / recharge
        Negative = pumping
        """

        W = np.zeros((model.nx, model.ny), dtype=float)
        cell_area = model.dx * model.dy

        # Wells
        for well in model.wells:
            i, j = well.cell
            # Convert m³/s → m/s (flux per area)
            W[i, j] -= well.Q / cell_area

        # Recharge (already m/s)
        if getattr(model, "recharge", None) is not None:
            W += model.recharge

        return W
