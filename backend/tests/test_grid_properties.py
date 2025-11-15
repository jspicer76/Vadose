import numpy as np
from backend.core.grid import Grid
from backend.core.properties import AquiferProperties


dx = np.array([10, 20, 30])
dy = np.array([10, 10])
grid = Grid(dx=dx, dy=dy)

Kx = Ky = Kz = np.ones((1, 3, 2)) * 5.0
thickness = np.array([50.0])

props = AquiferProperties(Kx=Kx, Ky=Ky, Kz=Kz,
                          thickness=thickness,
                          Sy=np.array([0.15]),
                          Ss=np.array([1e-5]),
                          porosity=np.array([0.25]),
                          confined=False)

print("Grid centers:", grid.cell_center(0,0))
print("Neighbors:", grid.neighbors(1,0))
print("Tx:", props.transmissivity_x(0,1,1))
