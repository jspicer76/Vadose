# backend/solvers/transient/budget.py

import numpy as np

class BudgetEngine:
    """
    Computes MODFLOW-style mass balance components and
    records per-time-step water budget for transient models.

    Components:
        - Wells
        - Recharge
        - Boundaries
        - Storage change
    """

    def __init__(self, model):
        self.model = model
        self.records = []   # store per-step budget info

    # --------------------------------------------------------------
    # WELL FLUXES
    # --------------------------------------------------------------
    def compute_wells(self):
        """Return (inflow, outflow) from wells. Positive = into model."""
        Q_in = 0.0
        Q_out = 0.0

        wells = getattr(self.model, "wells", [])
        for w in wells:
            if w.Q >= 0:     # positive Q = injection = inflow
                Q_in += w.Q
            else:            # negative Q = pumping = outflow
                Q_out += -w.Q

        return Q_in, Q_out

    # --------------------------------------------------------------
    # RECHARGE FLUX
    # --------------------------------------------------------------
    def compute_recharge(self, R_field):
        """
        Compute recharge inflow (always positive).
        R_field expected to be a 2D array matching grid.
        """
        if R_field is None:
            return 0.0, 0.0

        cell_area = self.model.dx * self.model.dy
        total_R = np.sum(R_field) * cell_area
        return total_R, 0.0

    # --------------------------------------------------------------
    # STORAGE CHANGE
    # --------------------------------------------------------------
        # --------------------------------------------------------------
    # STORAGE CHANGE
    # --------------------------------------------------------------
    def compute_storage_change(self, h_old_flat, h_new_flat, dt):
        """
        Compute volumetric storage change rate (L^3/T).

        Positive = water entering model  
        Negative = water leaving model
        """

        nx, ny = self.model.nx, self.model.ny

        # 1. Get storage vector from Storage module
        #    NOTE: We call Storage(model).vector() on demand.
        from backend.solvers.transient.storage import Storage
        S_vec = Storage(self.model).vector()      # shape (N,)

        # 2. Reshape heads
        h_old = h_old_flat.flatten()
        h_new = h_new_flat.flatten()

        # 3. Head change
        dh = h_new - h_old

        # 4. Storage is S * dh * area
        cell_area = self.model.dx * self.model.dy
        dV = np.sum(S_vec * dh) * cell_area

        # 5. Convert to flux (divide by dt)
        q_storage = dV / dt

        # Split into inflow/outflow
        if q_storage >= 0:
            return q_storage, 0.0
        else:
            return 0.0, -q_storage

    # --------------------------------------------------------------
    # BOUNDARY FLUXES (General Head, River, Drain)
    # --------------------------------------------------------------
    def compute_bc_fluxes(self, h):
        """Return (inflow, outflow) for boundary conditions."""
        bc_list = getattr(self.model, "boundary_conditions", [])
        nx, ny = self.model.nx, self.model.ny

        Q_in = 0.0
        Q_out = 0.0

        for bc in bc_list:
            if not hasattr(bc, "C"):
                continue

            for idx in bc.cells:
                i = idx // ny
                j = idx % ny

                h_cell = h[i, j]
                h_bc = getattr(bc, "hb", getattr(bc, "stage", 0.0))

                q = bc.C * (h_bc - h_cell) * (self.model.dx * self.model.dy)

                if q >= 0:
                    Q_in += q
                else:
                    Q_out += -q

        return Q_in, Q_out

    # --------------------------------------------------------------
    # NEW: RECORD A TIME STEP
    # --------------------------------------------------------------
    def record_step(self, step, t, h_new_flat, h_old_flat, dt):
        """
        Records full water budget for a transient step.
        """

        # reshape heads into (nx, ny)
        nx, ny = self.model.nx, self.model.ny
        h_new = h_new_flat.reshape((nx, ny))
        h_old = h_old_flat.reshape((nx, ny))

        
        # Recharge field at time t
        if hasattr(self.model, "get_recharge_field"):
            R_field = self.model.get_recharge_field(t)
        else:
            R_field = None

        # Compute components
        well_in,   well_out   = self.compute_wells()
        rech_in,   rech_out   = self.compute_recharge(R_field)
        bc_in,     bc_out     = self.compute_bc_fluxes(h_new)
        stor_in,   stor_out   = self.compute_storage_change(
            h_old_flat, h_new_flat, dt
        )

        total_in  = well_in + rech_in + bc_in + stor_in
        total_out = well_out + rech_out + bc_out + stor_out

        imbalance = total_in - total_out
        pct_error = 0.0 if total_in == 0 else abs(imbalance) / abs(total_in) * 100

        # store record
        self.records.append({
            "step": step,
            "time": t,
            "dt": dt,
            "inflow": {
                "wells": well_in,
                "recharge": rech_in,
                "boundaries": bc_in,
                "storage": stor_in,
            },
            "outflow": {
                "wells": well_out,
                "recharge": rech_out,
                "boundaries": bc_out,
                "storage": stor_out,
            },
            "total_in": total_in,
            "total_out": total_out,
            "imbalance": imbalance,
            "pct_error": pct_error,
        })

    # --------------------------------------------------------------
    # Full summary for debugging or export
    # --------------------------------------------------------------
    def summarize(self):
        return self.records
