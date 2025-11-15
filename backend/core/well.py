from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Well:
    """
    Represents a pumping or injection well.
    This class is solver-agnostic and contains physical well properties only.
    """

    name: str
    x: float                     # x-coordinate (model units)
    y: float                     # y-coordinate (model units)
    rate: float                  # Pumping rate: negative = pumping, positive = injection
    radius: float = 0.5          # Well radius (not yet used in steady-state solution)
    screen_top: Optional[float] = None      # Elevation of top of screened interval
    screen_bottom: Optional[float] = None   # Elevation of bottom of screened interval

    # Will be set by the model after grid assignment
    cell_index: Optional[Tuple[int, int]] = None

    def assign_to_grid(self, grid):
        """
        Determine which (i,j) cell the well belongs to based on (x,y).
        """
        self.cell_index = grid.get_cell_index(self.x, self.y)

    def get_rhs_contribution(self, nx, ny):
        """
        Return a vector (flattened) containing the well’s volumetric rate contribution.
        Only the cell containing the well receives the pumping rate.

        Parameters:
        - nx, ny: grid dimensions

        Output:
        - q: flattened array of size (nx*ny,)
        """
        import numpy as np

        q = np.zeros(nx * ny)

        if self.cell_index is None:
            raise RuntimeError(f"Well '{self.name}' is not assigned to a grid cell.")

        i, j = self.cell_index
        idx = j * nx + i     # row-major flattening

        q[idx] += self.rate

        return q
