import numpy as np


class BudgetEngine:
    """
    Lightweight water-budget tracker used by the transient solver.

    The goal is not to provide a perfect MODFLOW-style budget but to
    offer enough diagnostics to spot obvious errors (e.g., recharge not
    applied, storage term exploding, etc.).
    """

    def __init__(self, model):
        self.model = model
        self.cell_area = model.dx * model.dy

    # ------------------------------------------------------------------
    def _recharge_volume(self, t, dt):
        """
        Return recharge addition/removal (m^3) over the time step.
        """
        R = None
        if hasattr(self.model, "get_recharge_field"):
            R = self.model.get_recharge_field(t)

        if R is None:
            return 0.0

        # Sum of fluxes (L/T) * area * dt → volume
        flux_total = np.sum(np.asarray(R, dtype=float))
        return flux_total * self.cell_area * dt

    # ------------------------------------------------------------------
    def _well_volume(self, dt):
        """
        Wells are provided as volumetric rates (m^3/s). Positive rates
        correspond to extraction in our convention.
        """
        if not getattr(self.model, "wells", None):
            return 0.0

        total = 0.0
        for well in self.model.wells:
            total -= well.Q * dt  # subtract because SourceSink uses -Q/area
        return total

    # ------------------------------------------------------------------
    def _storage_change(self, h_old, h_new):
        """
        Storage coefficient (C) is defined per cell. Multiply by head
        change (m) and area (m^2) to obtain volume (m^3).
        """
        storage = getattr(self.model, "storage", None)
        if storage is None:
            return 0.0
        dh = np.asarray(h_new, dtype=float) - np.asarray(h_old, dtype=float)
        return float(np.sum(storage * dh) * self.cell_area)

    # ------------------------------------------------------------------
    def summarize(self, step, t, dt, h_old, h_new):
        """
        Produce a short human-readable balance string.
        """
        rech = self._recharge_volume(t, dt)
        wells = self._well_volume(dt)
        storage = self._storage_change(h_old, h_new)

        # Positive recharge/well values mean net addition to storage
        net_in = rech + wells
        residual = net_in - storage

        return (
            f"[step {step:04d} t={t:.1f}s] "
            f"Recharge={rech:.3e} m^3, "
            f"Wells={wells:.3e} m^3, "
            f"ΔStorage={storage:.3e} m^3, "
            f"Residual={residual:.3e} m^3"
        )
