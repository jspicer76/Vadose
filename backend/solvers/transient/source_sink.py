# backend/solvers/transient/source_sink.py

import numpy as np

class SourceSink:
    """
    Handles wells, injections, recharge-as-sources, etc.
    """

    @staticmethod
    def build_W_vector(model, t):
        """
        Returns W array same shape as model.h of volumetric flux per cell.
        + positive for injection/recharge
        - negative for pumping
        """

        W = np.zeros((model.nx, model.ny))

        # Wells
        for well in model.wells:
            i, j = well.cell
            W[i, j] -= well.Q  # pumping negative sign convention

        # Recharge (MODFLOW-style RCH)
        if getattr(model, "recharge", None) is not None:
            W += model.recharge  # distributed recharge array


        return W
