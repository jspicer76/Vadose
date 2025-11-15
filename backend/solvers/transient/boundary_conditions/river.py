# backend/solvers/transient/boundary_conditions/river.py

class RiverBC:
    """
    River boundary with riverbed resistance.
    
    q = C (h_river - h_cell)
    """

    def __init__(self, cells, conductance, river_stage):
        self.cells = cells
        self.C = conductance
        self.stage = river_stage

    def apply(self, A, b):
        for idx in self.cells:
            A[idx, idx] += self.C
            b[idx] += self.C * self.stage
        return A, b
