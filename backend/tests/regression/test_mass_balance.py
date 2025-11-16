from backend.solvers.transient.transient_solver import run_transient_solver

def test_mass_balance_small_error():
    nx = 21

    wells = [{"i":10, "j":10, "Q":0.01}]
    config = {
        "nx": nx, "ny": nx,
        "dx": 10, "dy": 10,
        "dt": 10, "steps": 20,
        "T": 100.0, "S": 0.001,
        "initial_head": 10.0,
        "pumping_wells": wells,
    }

    heads = run_transient_solver(config)
    # If solver mass balance is good, it should not crash or produce bizarre heads.
    final = heads[-1]
    assert final.min() > 0
    assert final.max() < 11
