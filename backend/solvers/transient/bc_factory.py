# backend/solvers/transient/bc_factory.py

"""
BCFactory: Converts simple config dictionaries into boundary condition
class instances used by the transient solver.

Example configurations:

    {"type": "dirichlet", "cells": [...], "value": 100}

    {"type": "ghb", "cells": [...], "stage": 10, "conductance": 5e-5}

    {"type": "river", "cells": [...], "stage": 8.5, "conductance": 1e-4}

    {"type": "flux", "cells": [...], "flux": -1e-6}
"""

from typing import Dict, Any

# BC class imports
from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC
from backend.solvers.transient.boundary_conditions.neumann import NeumannBC
from backend.solvers.transient.boundary_conditions.general_head import GeneralHeadBC
from backend.solvers.transient.boundary_conditions.river import RiverBC
from backend.solvers.transient.boundary_conditions.recharge import RechargeBC
from backend.solvers.transient.boundary_conditions.theis_dirichlet import TheisDirichletBC
from backend.solvers.transient.boundary_conditions.flux_bc import FluxBC


class BCFactory:
    """
    Factory class to create BC objects from simple dictionaries.

    NOTE:
    - For boundaries requiring `model` (FluxBC), the model must
      be injected later by TransientModel or TransientSolver.
    """

    @staticmethod
    def create(bc_conf: Dict[str, Any], model=None):
        """
        Convert a BC config dict into a BC instance.

        If a BC requires `model` (FluxBC), the caller must pass it.
        """
        if "type" not in bc_conf:
            raise ValueError("BC config missing 'type' key")

        bc_type = bc_conf["type"].lower()

        # ----------------------------------------------------------
        # Dirichlet (fixed head)
        # ----------------------------------------------------------
        if bc_type == "dirichlet":
            return DirichletBC(
                cells=bc_conf["cells"],
                head_value=bc_conf["value"]
            )

        # ----------------------------------------------------------
        # Neumann BC (legacy): same as flux but NO volume conversion
        # ----------------------------------------------------------
        if bc_type == "neumann":
            return NeumannBC(
                cells=bc_conf["cells"],
                flux_value=bc_conf["value"]
            )

        # ----------------------------------------------------------
        # General-head BC (MODFLOW GHB)
        # ----------------------------------------------------------
        if bc_type == "ghb":
            C = bc_conf.get("conductance", bc_conf.get("C"))
            stage = bc_conf.get("stage")
            if C is None or stage is None:
                raise ValueError("GHB requires 'stage' and 'conductance'")
            return GeneralHeadBC(
                cells=bc_conf["cells"],
                conductance=C,
                boundary_head=stage
            )

        # ----------------------------------------------------------
        # River BC
        # ----------------------------------------------------------
        if bc_type == "river":
            return RiverBC(
                cells=bc_conf["cells"],
                stage=bc_conf["stage"],
                conductance=bc_conf["conductance"]
            )

        # ----------------------------------------------------------
        # Recharge BC (specialized)
        # ----------------------------------------------------------
        if bc_type == "recharge":
            return RechargeBC(
                cells=bc_conf["cells"],
                rate=bc_conf["rate"]
            )

        # ----------------------------------------------------------
        # Theis infinite-extent perimeter BC
        # ----------------------------------------------------------
        if bc_type == "theis":
            return TheisDirichletBC(
                cells=bc_conf["cells"],
                cell_coords=bc_conf["cell_coords"],
                wells=bc_conf["wells"],
                T=bc_conf["T"],
                S=bc_conf["S"],
                dx=bc_conf["dx"],
                dy=bc_conf["dy"],
                head0=bc_conf["head0"]
            )

        # ----------------------------------------------------------
        # NEW: Specified FLUX BC  (this is your M.1f Step 2)
        # ----------------------------------------------------------
        if bc_type == "flux":
            if model is None:
                raise ValueError("FluxBC requires model reference")
            return FluxBC(
                cells=bc_conf["cells"],
                flux=bc_conf["flux"],   # scalar, array, or callable
                model=model
            )

        # ----------------------------------------------------------
        # Unknown BC type
        # ----------------------------------------------------------
        raise ValueError(f"Unknown BC type '{bc_type}'")
