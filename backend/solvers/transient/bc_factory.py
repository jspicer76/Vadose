# backend/solvers/transient/bc_factory.py

"""
BCFactory: Converts simple config dictionaries into boundary condition
class instances used by the transient solver.

This allows regression tests and solver configs to specify BCs like:

    {"type": "dirichlet", "cells": [...], "value": 100}

    {"type": "ghb", "cells": [...], "stage": 10.0, "C": 5e-5}

    {"type": "river", "cells": [...], "stage": 9.0, "conductance": 1e-3}

"""

from typing import Dict, Any

# Import all BC classes your solver supports
from backend.solvers.transient.boundary_conditions.dirichlet import DirichletBC
from backend.solvers.transient.boundary_conditions.neumann import NeumannBC
from backend.solvers.transient.boundary_conditions.general_head import GeneralHeadBC
from backend.solvers.transient.boundary_conditions.river import RiverBC
from backend.solvers.transient.boundary_conditions.recharge import RechargeBC
from backend.solvers.transient.boundary_conditions.theis_dirichlet import TheisDirichletBC


class BCFactory:
    """Factory class to create boundary condition objects from dict configs."""

    @staticmethod
    def create(bc_conf: Dict[str, Any]):
        """
        Accepts a dict from config or regression tests and returns
        an instantiated boundary condition object.

        Example input:
            {"type": "dirichlet", "cells": [...], "value": 100}

            {"type": "ghb", "cells": [...], "stage": 10, "C": 1e-3}

        """
        if "type" not in bc_conf:
            raise ValueError(f"Boundary condition missing 'type': {bc_conf}")

        bc_type = bc_conf["type"].lower()

        # ------------------------------------------------------------------
        #                       DIRICHLET (fixed head)
        # ------------------------------------------------------------------
        if bc_type == "dirichlet":
            return DirichletBC(
                cells=bc_conf["cells"],
                head_value=bc_conf["value"]
            )

        # ------------------------------------------------------------------
        #                       NEUMANN (specified flux)
        # ------------------------------------------------------------------
        if bc_type == "neumann":
            return NeumannBC(
                cells=bc_conf["cells"],
                flux_value=bc_conf["value"]
            )

        # ------------------------------------------------------------------
        #         GENERAL HEAD BOUNDARY (MODFLOW GHB-like)
        # ------------------------------------------------------------------
        if bc_type == "ghb":
            conductance = bc_conf.get("conductance", bc_conf.get("C"))
            stage = bc_conf.get("stage")
            if conductance is None or stage is None:
                raise ValueError(
                    f"GHB requires 'stage' and 'C/conductance': {bc_conf}"
                )
            return GeneralHeadBC(
                cells=bc_conf["cells"],
                conductance=conductance,
                boundary_head=stage
            )

        # ------------------------------------------------------------------
        #                            RIVER BC
        # ------------------------------------------------------------------
        if bc_type == "river":
            return RiverBC(
                cells=bc_conf["cells"],
                stage=bc_conf["stage"],
                conductance=bc_conf["conductance"]
            )

        # ------------------------------------------------------------------
        #                        RECHARGE BC
        # ------------------------------------------------------------------
        if bc_type == "recharge":
            return RechargeBC(
                cells=bc_conf["cells"],
                rate=bc_conf["rate"]
            )

        # ------------------------------------------------------------------
        #                        THEIS-DIRICHLET BC
        #    (Used by your perimeter infinite aquifer test harness)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        #                     Unsupported BC type
        # ------------------------------------------------------------------
        raise ValueError(f"Unknown boundary condition type '{bc_type}' in {bc_conf}")
