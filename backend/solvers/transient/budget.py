# backend/solvers/transient/budget.py

import numpy as np

class BudgetEngine:
    """
    Computes MODFLOW-style mass balance components
    and generates formatted summary tables.

    Components:
        - Wells
        - Recharge
        - Constant / General Head boundaries
        - Storage
    """

    def __init__(self, model):
        self.model = model

    # --------------------------------------------------------------
    # WELL FLUXES
    # --------------------------------------------------------------
    def compute_wells(self):
        """Return (inflow, outflow) from wells."""
        Q_in = 0.0
        Q_out = 0.0

        wells = getattr(self.model, "wells", [])
        for w in wells:
            if w.Q >= 0:      # pumping draws from system
                Q_out += w.Q
            else:             # injection adds to system
                Q_in += -w.Q

        return Q_in, Q_out

    # --------------------------------------------------------------
    # RECHARGE
    # --------------------------------------------------------------
    def compute_recharge(self):
        """Compute recharge volume (always inflow)."""
        rec = getattr(self.model, "recharge", None)
        if isinstance(rec, np.ndarray):
            cell_area = self.model.dx * self.model.dy
            return np.sum(rec) * cell_area, 0.0
        return 0.0, 0.0

    # --------------------------------------------------------------
    # STORAGE CHANGE
    # --------------------------------------------------------------
    def compute_storage_change(self, h_old, h_new, dt):
        """
        Compute volumetric storage change (not a flux rate):
            dV = S * Δh * area

        MODFLOW convention:
        - Positive = water entering model (rise in head)
        - Negative = water leaving model (decline in head)
        """
        S = self.model.storage              # 2D array
        dh = h_new - h_old                  # Δh
        active = self.model.active          # active mask

        cell_area = self.model.dx * self.model.dy

        # Only active cells contribute and scale to volumetric rate
        dV = np.sum(S[active] * dh[active]) * cell_area
        q_storage = dV / dt

        # Split into inflow / outflow components
        if q_storage >= 0:
            return q_storage, 0.0
        else:
            return 0.0, -q_storage


    # --------------------------------------------------------------
    # BOUNDARY FLUXES
    # --------------------------------------------------------------
    def compute_bc_fluxes(self, h):
        """
        Compute fluxes for boundaries that use conductance:
        e.g., General-Head, River, Drain.
        """
        Q_in = 0.0
        Q_out = 0.0

        bcs = getattr(self.model, "boundary_conditions", [])
        nx, ny = self.model.nx, self.model.ny

        for bc in bcs:
            if hasattr(bc, "C"):  # Conductance exists
                for idx in bc.cells:
                    i = idx // ny
                    j = idx % ny

                    h_cell = h[i, j]
                    h_bc = bc.hb if hasattr(bc, "hb") else getattr(bc, "stage", 0.0)

                    q = bc.C * (h_bc - h_cell) * (self.model.dx * self.model.dy)

                    if q >= 0:
                        Q_in += q
                    else:
                        Q_out += -q

        return Q_in, Q_out

    # --------------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------------
    def summarize(self, step, t, dt, h_old, h_new):
        """Build MODFLOW-style summary table as a string."""

        well_in, well_out     = self.compute_wells()
        rech_in, rech_out     = self.compute_recharge()
        stor_in, stor_out     = self.compute_storage_change(h_old, h_new, dt)
        bc_in, bc_out         = self.compute_bc_fluxes(h_new)

        total_in  = well_in + rech_in + bc_in + stor_in
        total_out = well_out + rech_out + bc_out + stor_out

        error = 0.0 if total_in == 0 else abs(total_in - total_out) / total_in * 100

        lines = []
        lines.append("\n===============================================================")
        lines.append("                    VADOSE MASS BALANCE SUMMARY                ")
        lines.append("===============================================================")
        lines.append(f"Time step:        {step}")
        lines.append(f"Simulation time:  {t:.2f} s")
        lines.append(f"Δt:               {dt:.2f} s")
        lines.append("---------------------------------------------------------------")
        lines.append("Component                  Inflow         Outflow         Net")
        lines.append("---------------------------------------------------------------")
        lines.append(f"Wells                     {well_in:10.3f}   {well_out:10.3f}   {well_in - well_out:10.3f}")
        lines.append(f"Recharge                  {rech_in:10.3f}   {rech_out:10.3f}   {rech_in - rech_out:10.3f}")
        lines.append(f"BC Conductance Flux       {bc_in:10.3f}   {bc_out:10.3f}   {bc_in - bc_out:10.3f}")
        lines.append(f"Storage Change            {stor_in:10.3f}   {stor_out:10.3f}   {stor_in - stor_out:10.3f}")
        lines.append("---------------------------------------------------------------")
        lines.append(f"TOTALS                    {total_in:10.3f}   {total_out:10.3f}   {total_in - total_out:10.3f}")
        lines.append(f"Mass Balance Error (%)     {error:6.3f}%")
        lines.append("===============================================================\n")

        return "\n".join(lines)
