import numpy as np
import matplotlib.pyplot as plt

def plot_head_contour(h, dx, dy, title="Head Contour"):
    """
    Plot a 2D contour map of head.

    Parameters
    ----------
    h : array (nx, ny)
    dx, dy : float
    """

    nx, ny = h.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(8,6))
    cs = plt.contourf(X, Y, h, 20, cmap='viridis')
    plt.colorbar(cs, label="Head (m)")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.show()
