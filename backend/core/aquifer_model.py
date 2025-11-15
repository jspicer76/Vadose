from dataclasses import dataclass, field
from typing import List, Optional

from .grid import ModelGrid
from .layers import Layer
from .wells import Well
from .boundaries import BoundaryCondition
from .parameters import SolverParameters


@dataclass
class AquiferModel:
    """
    Top-level container for all groundwater model data.

    This is what steady-state and transient solvers will operate on.
    """
    name: str
    grid: ModelGrid
    layers: List[Layer] = field(default_factory=list)
    wells: List[Well] = field(default_factory=list)
    boundaries: List[BoundaryCondition] = field(default_factory=list)
    solver_params: SolverParameters = field(default_factory=SolverParameters)

    def summary(self) -> str:
        """Return a short text summary of the model for debugging."""
        return (
            f"AquiferModel(name={self.name!r}, "
            f"cells={self.grid.n_cells()}, "
            f"layers={len(self.layers)}, "
            f"wells={len(self.wells)}, "
            f"boundaries={len(self.boundaries)})"
        )
