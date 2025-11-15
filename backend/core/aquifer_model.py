from dataclasses import dataclass, field
from backend.core.grid import Grid
from backend.core.properties import AquiferProperties
from backend.core.well import Well
from backend.solvers.steady_state.interface import SteadyStateSolver


@dataclass
class AquiferModel:
    name: str
    grid: Grid
    properties: AquiferProperties
    wells: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)

    # -----------------------------------------------
    # Hydraulic property access
    # -----------------------------------------------
    def transmissivity_tensor(self, i, j, k=0):
        Tx = self.properties.transmissivity_x(k, i, j)
        Ty = self.properties.transmissivity_y(k, i, j)
        return Tx, Ty

    def storage(self, i, j, k=0):
        if self.properties.confined:
            return self.properties.Ss[k] * self.properties.thickness[k]
        else:
            return self.properties.Sy[k]

    # -----------------------------------------------
    # Steady-state solver hook
    # -----------------------------------------------
    def solve_steady_state(self, method="direct"):
        solver = SteadyStateSolver()
        result = solver.solve(self, method=method)
        self.last_solution = result
        return result

    def run(self, method="direct"):
        return self.solve_steady_state(method)
