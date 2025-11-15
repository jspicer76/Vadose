# backend/solvers/transient/boundary_conditions/dirichlet.py

class DirichletBC:
    """
    Fixed-head boundary condition.
    """

    def __init__(self, cells, head_value):
        self.cells = cells
        self.head_value = head_value

    def apply(self, A, b):
        """
        Modifies A and b to enforce h(i,j) = head_value.
        """
        for idx in self.cells:
            A[idx, :] = 0.0
            A[idx, idx] = 1.0
            b[idx] = self.head_value
        return A, b
