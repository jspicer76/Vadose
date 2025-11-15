from dataclasses import dataclass, field
import numpy as np

@dataclass
class Grid:
    dx: np.ndarray    # array of cell widths in x-direction
    dy: np.ndarray    # array of cell widths in y-direction
    nlay: int = 1     # number of vertical layers

    def __post_init__(self):
        self.nx = len(self.dx)
        self.ny = len(self.dy)

        # Compute coordinates of cell centers
        self.x_centers = np.cumsum(self.dx) - self.dx / 2
        self.y_centers = np.cumsum(self.dy) - self.dy / 2

        # Face positions (x0, x1, x2 …)
        self.x_faces = np.concatenate(([0], np.cumsum(self.dx)))
        self.y_faces = np.concatenate(([0], np.cumsum(self.dy)))

    def cell_center(self, i, j):
        """Return (x,y) location of cell center."""
        return (self.x_centers[i], self.y_centers[j])

    def cell_size(self, i, j):
        """Return (dx,dy) of a cell."""
        return (self.dx[i], self.dy[j])

    def neighbors(self, i, j):
        """Return neighbors for finite-difference stencil."""
        neigh = {}
        if i > 0:
            neigh["west"] = (i - 1, j)
        if i < self.nx - 1:
            neigh["east"] = (i + 1, j)
        if j > 0:
            neigh["south"] = (i, j - 1)
        if j < self.ny - 1:
            neigh["north"] = (i, j + 1)
        return neigh

    def get_cell_index(self, x, y):
        """Return (i,j) indices for a given spatial coordinate."""
        i = np.searchsorted(self.x_faces, x) - 1
        j = np.searchsorted(self.y_faces, y) - 1

        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            raise ValueError("Point outside grid domain")

        return (i, j)
