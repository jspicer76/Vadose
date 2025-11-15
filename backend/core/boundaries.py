from dataclasses import dataclass
from typing import Literal


BoundaryType = Literal["specified_head", "specified_flux", "no_flow", "river"]


@dataclass
class BoundaryCondition:
    """
    Generic boundary condition.

    Later we'll extend this or add subclasses for rivers, general head, etc.
    """
    btype: BoundaryType
    value: float  # meaning depends on type (head, flux, etc.)
    x: float
    y: float
