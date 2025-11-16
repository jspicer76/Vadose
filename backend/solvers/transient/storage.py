# backend/solvers/transient/storage.py

import numpy as np

class Storage:
    """
    Storage coefficient builder for transient groundwater flow.

    New architecture:
      - Accepts model in constructor
      - Provides .vector() returning flattened S array
      - Uses 'S' for confined, 'Sy' for unconfined
      - Supports inactive cells
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    def vector(self):
        """
        Returns the storage vector S for each active cell in the model,
        flattened into a 1D array of length nx*ny.

        Confined:
            S = storativity (dimensionless)
        Unconfined:
            Sy = specific yield (dimensionless)

        Inactive cells receive zero storage.
        """
        m = self.model
        nx, ny = m.nx, m.ny

        # Determine active mask (default: all active)
        active = getattr(m, "active", np.ones((nx, ny), dtype=bool))

        # Determine S field based on aquifer type
        if m.aquifer_type == "confined":
            S_field = np.full((nx, ny), float(m.S))
        elif m.aquifer_type == "unconfined":
            S_field = np.full((nx, ny), float(m.Sy))
        else:
            raise ValueError(f"Unknown aquifer type: {m.aquifer_type}")

        # Zero out inactive cells
        S_field = np.where(active, S_field, 0.0)

        # Flatten to match solver expectations
        return S_field.flatten()
