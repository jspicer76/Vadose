from dataclasses import dataclass
from typing import Optional


@dataclass
class Well:
    """
    Pumping or injection well.

    Q > 0 for injection, Q < 0 for pumping (following typical sign convention).
    """
    name: str
    x: float
    y: float
    Q: float  # volumetric rate (L^3/T)
    screen_top: Optional[float] = None
    screen_bottom: Optional[float] = None
