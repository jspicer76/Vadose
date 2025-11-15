# backend/solvers/transient/boundary_conditions/general_head.py

class GeneralHeadBC:
    """
    General-head boundary (conductance boundary).
    q = C (h_b - h_cell)
    """

    def __init__(self, cells, conductance, boundary_head):
        self.cells = cells
        self.C = conductance
        self.hb = boundary_head

    def apply(self, A, b):
        for idx in self.cells:
            A[idx, idx] += self.C
            b[idx] += self.C * self.hb
        return A, b
