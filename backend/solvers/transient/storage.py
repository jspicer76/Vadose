# backend/solvers/transient/storage.py

import numpy as np

class Storage:
    """
    Computes storage values (S for confined, Sy for unconfined).
    """

    @staticmethod
    def compute_storage(model, h):
        """
        Returns S array matching model grid.
        """
        if model.aquifer_type == "confined":
            return np.full((model.nx, model.ny), model.S)

        elif model.aquifer_type == "unconfined":
            # linearized Boussinesq (initial h_old thickness)
            return np.full((model.nx, model.ny), model.Sy)

        else:
            raise ValueError("Unknown aquifer type")
