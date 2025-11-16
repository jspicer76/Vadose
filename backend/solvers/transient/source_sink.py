# backend/solvers/transient/source_sink.py

import numpy as np


class SourceSink:
    """
    Builds volumetric source/sink flux fields (per unit area)
    for the transient groundwater solver.

    - Wells:      Q [L^3/T] → q = Q / Area  [L/T]
    - Recharge:   R(x,y,t) already in flux form [L/T]

    Positive flux  = injection / recharge
    Negative flux  = extraction (pumping)
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    # Wells contribution
    # ------------------------------------------------------------------
    def wells_flux(self, t):
        """
        Returns wells as flux per unit area (L/T),
        shape = (nx, ny).
        """
        model = self.model
        W = np.zeros((model.nx, model.ny), dtype=float)

        if not model.wells:
            return W

        cell_area = model.dx * model.dy

        for well in model.wells:
            i, j = well.cell
            # Q is volumetric rate (m³/s or ft³/s)
            # Convert to flux per unit area
            W[i, j] += -well.Q / cell_area   # pumping is negative

        return W

    # ------------------------------------------------------------------
    # Recharge contribution
    # ------------------------------------------------------------------
    def recharge_flux(self, t):
        """
        Returns recharge field [L/T], shape = (nx, ny).

        Uses TransientModel.get_recharge_field(t) which supports:
          - scalar
          - 2D array
          - flat array
          - callable(t)
        """
        R = self.model.get_recharge_field(t)
        if R is None:
            return np.zeros((self.model.nx, self.model.ny), dtype=float)

        return R.astype(float)

    # ------------------------------------------------------------------
    # Combined W-field
    # ------------------------------------------------------------------
    def combined_flux(self, t):
        """
        Returns full W-field: wells + recharge, flux per area [L/T].
        Equivalent to MODFLOW "source term".
        """
        return self.wells_flux(t) + self.recharge_flux(t)

    # ------------------------------------------------------------------
    # Backward compatibility method (old name used in solver)
    # ------------------------------------------------------------------
    @staticmethod
    def build_W_vector(model, t):
        """
        Legacy static API used by old solver versions.
        Returns W(x,y) [L/T].

        This redirects into the new SourceSink class.
        """
        ss = SourceSink(model)
        return ss.combined_flux(t)
