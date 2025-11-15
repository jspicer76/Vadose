"""
Steady-state groundwater flow solver.

Eventually this will solve ∇·(K ∇h) = Q on the ModelGrid.
"""

from ..core.aquifer_model import AquiferModel


def solve_steady_state(model: AquiferModel):
    """
    Placeholder steady-state solver.

    For M.1b we just exercise the structure and return a dummy result.
    """
    # Later: assemble coefficient matrix, apply BCs, solve for h.
    return {
        "status": "not_implemented",
        "model_summary": model.summary(),
    }
