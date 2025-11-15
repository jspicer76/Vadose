# backend/solvers/transient/budget.py

import numpy as np

class BudgetEngine:
    """
    Computes MODFLOW-style mass balance components
    and generates formatted summary tables.

    Components:
        - Constant head boundaries
        - General head boundaries
        - River leakage
        - Recharge
        - Wells
        - Storage change
    """

    def __init__(self, model):
        self.model = model

    # --------------------------------------------------------------
    # FLUX CALCULATIONS
    # --------------------------------------------------------------

    def compute_wells(self):
        """Return (inflow, outflow) from wells."""
        Q_in = 0.0
        Q_out = 0.0

        if not hasattr(self.model, "wells"):
            return Q_in, Q_out

        for w in self.model.wells:
            if w.Q > 0:
                Q_in += w.Q
            else:
                Q_out += -w.Q

        return Q_in, Q_out

    def compute_recharge(self):
        """Compute volumetric recharge (positive only)."""
        rec = getattr(self.model, "recharge", None)
        if isinstance(rec, np.ndarray):
            vol = np.sum(rec) * self.model.dx * self.model.dy
            return vol, 0.0
        return 0.0, 0.0

    def compute_storage_change(self, h_old, h_new, dt):
        """Compute change in aquifer storage."""
        S = self.model.storage
        dV = np.sum(S * (h_new - h_old) / dt) * self.model.dx * self.model.dy

        if dV >= 0:
            return dV, 0.0
        else:
            return 0.0, -dV

    def compute_bc_fluxes(self, h):
        """
        Compute fluxes from boundary conditions that rely
        on conductance: GHB, River, etc.

        Dirichlet BC fluxes will be handled separately because
        they overwrite A/b but do not contribute a natural flux.
        """

        Q_in = 0.0
        Q_out = 0.0

        if not hasattr(self.model, "boundary_conditions"):
            return Q_in, Q_out

        nx, ny = self.model.nx, self.model.ny

        for bc in self.model.boundary_conditions:
            if hasattr(bc, "C"):  # Conductance-based BC
                for idx in bc.cells:

                    i = idx // ny
                    j = idx % ny

                    h_cell = h[i, j]
                    h_bc = bc.hb if hasattr(bc, "hb") else getattr(bc, "stage", 0.0)
                    C = bc.C

                    q = C * (h_bc - h_cell) * self.model.dx * self.model.dy

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

        # Wells
        well_in, well_out = self.compute_wells()

        # Recharge
        rech_in, rech_out = self.compute_recharge()

        # Storage
        stor_in, stor_out = self.compute_storage_change(h_old, h_new, dt)

        # BC Fluxes
        bc_in, bc_out = self.compute_bc_fluxes(h_new)

        # Totals
        total_in = well_in + rech_in + bc_in + stor_in
        total_out = well_out + rech_out + bc_out + stor_out

        error = 0.0 if total_in == 0 else abs(total_in - total_out) / total_in * 100

        # Build formatted table
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
