# backend/visualization/plot_profiles.py

import numpy as np
import matplotlib.pyplot as plt

def plot_head_profile(heads, dx, times, indices=None):
    """
    Plot head vs. distance for selected time steps.

    Parameters
    ----------
    heads : array, shape (nt, nx, ny)
        Time series of head fields.
    dx : float
        Cell size in x-direction.
    times : list
        List of time step indices to plot.
    indices : tuple
        (i, j) slice to extract. 
        If None and ny==1, assume 1D.
    """

    nt, nx, ny = heads.shape

    if ny > 1 and indices is None:
        raise ValueError("This is a 1D profile plot. Provide indices=(slice_j).")

    x = np.arange(nx) * dx

    plt.figure(figsize=(10, 6))

    for t in times:
        h = heads[t, :, 0] if ny == 1 else heads[t, :, indices[1]]
        plt.plot(x, h, label=f"t = {t}")

    plt.xlabel("Distance (m)")
    plt.ylabel("Head (m)")
    plt.title("Head Profile Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
