# backend/solvers/transient/boundary_conditions/neumann.py

class NeumannBC:
    """
    Fixed-flux boundary.
    Positive flux means flow into the domain.
    """

    def __init__(self, cells, flux):
        self.cells = cells
        self.flux = flux

    def apply(self, A, b):
        for idx in self.cells:
            b[idx] += self.flux
        return A, b
