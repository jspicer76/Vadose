from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelGrid:
    """
    Spatial grid for the aquifer model.

    For now we just store basic metadata:
      - nx, ny: number of cells in x and y
      - dx, dy: cell size (L)
    """
    nx: int
    ny: int
    dx: float
    dy: float

    def n_cells(self) -> int:
        """Return total number of cells."""
        return self.nx * self.ny
