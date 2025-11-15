# backend/visualization/animate_1d.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_head_1d(heads, dx, interval=100, save_path=None):
    """
    Create a 1D animation of head propagation over time.

    Parameters
    ----------
    heads : array (nt, nx, ny)
        Time series of heads.
    dx : float
        Spatial step.
    interval : int
        Delay between frames (ms).
    save_path : str or None
        If provided, saves .mp4 or .gif.
    """

    nt, nx, ny = heads.shape
    x = np.arange(nx) * dx

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(0, x[-1])
    ax.set_ylim(np.min(heads), np.max(heads))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Head (m)")
    ax.set_title("Transient 1D Head Propagation")

    def update(frame):
        y = heads[frame, :, 0]
        line.set_data(x, y)
        ax.set_title(f"timestep {frame}")
        return line,

    ani = FuncAnimation(fig, update, frames=nt, interval=interval, blit=True)

    if save_path:
        ani.save(save_path, fps=20)

    plt.show()
    return ani
