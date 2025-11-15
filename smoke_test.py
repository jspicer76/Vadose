from backend.core.grid import ModelGrid
from backend.core.layers import Layer
from backend.core.aquifer_model import AquiferModel
from backend.solvers.steady_state import solve_steady_state


def main():
    # Minimal dummy model just to prove imports and wiring
    grid = ModelGrid(nx=10, ny=10, dx=10.0, dy=10.0)
    layer = Layer(
        name="Layer 1",
        elevation_top=100.0,
        elevation_bottom=80.0,
        default_K=10.0,
        default_S=1e-4,
    )

    model = AquiferModel(
        name="Test Aquifer",
        grid=grid,
        layers=[layer],
    )

    result = solve_steady_state(model)
    print("Model summary:", model.summary())
    print("Solver result:", result)


if __name__ == "__main__":
    main()
