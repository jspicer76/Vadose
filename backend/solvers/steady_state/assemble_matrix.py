import numpy as np
from scipy.sparse import lil_matrix

def assemble_matrix(model, head_prev):
    """
    Assemble the finite-difference matrix (A) and RHS vector (b)
    for steady-state confined and unconfined flow.

    Confined:
        T = K * b

    Unconfined (BCF-style water-table correction):
        saturated_thickness = max(head - bottom, min_thick)
        T = K * saturated_thickness

    Boundary types supported:
        - "CONSTANT_HEAD"  (Dirichlet)
        - "NO_FLOW"        (Neumann)

    Wells:
        - included via Well.get_rhs_contribution()

    Returns:
        A (sparse matrix)
        b (RHS vector)
    """

    grid = model.grid
    props = model.properties

    nx, ny = grid.nx, grid.ny
    nlay = props.nlay
    assert nlay == 1, "Current M.1d implementation handles 1 layer only."

    # Flattened system size
    N = nx * ny

    # Initialize sparse matrix and RHS vector
    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N)

    # ---------- Compute saturated thickness ----------
    # Confined: thickness = layer thickness
    # Unconfined: adjust based on head (BCF method)

    thickness = np.zeros((nx, ny))
    base = model.properties.thickness[0]  # layer base thickness or unit thickness

    if model.properties.confined:
        for i in range(nx):
            for j in range(ny):
                thickness[i, j] = base
    else:
        # Unconfined
        bottom = np.zeros((nx, ny))  # future: integrate model-supplied bottom elevations
        min_thick = 0.1

        for i in range(nx):
            for j in range(ny):
                sat = max(head_prev[i, j] - bottom[i, j], min_thick)
                thickness[i, j] = sat

    # ---------- Build system ----------
    for j in range(ny):
        for i in range(nx):

            row = j * nx + i  # flatten index

            # Check for constant-head boundary
            ch = _constant_head_value(model, i, j)
            if ch is not None:
                A[row, row] = 1.0
                b[row] = ch
                continue

            # Compute transmissivities (USBR §5-10, Todd 2005)
            Kx = props.Kx[0, i, j]
            Ky = props.Ky[0, i, j]

            Ti = Kx * thickness[i, j]
            Tj = Ky * thickness[i, j]

            # Neighbor transmissivities
            tx_w = _trans_x(model, thickness, i, j, i-1, j)
            tx_e = _trans_x(model, thickness, i, j, i+1, j)
            ty_s = _trans_y(model, thickness, i, j, i, j-1)
            ty_n = _trans_y(model, thickness, i, j, i, j+1)

            # Diagonal coefficient
            A[row, row] = tx_w + tx_e + ty_s + ty_n

            # West neighbor
            if i > 0:
                col = row - 1
                A[row, col] = -tx_w
            # No-flow west boundary → skip (Neumann)

            # East neighbor
            if i < nx - 1:
                col = row + 1
                A[row, col] = -tx_e

            # South neighbor
            if j > 0:
                col = row - nx
                A[row, col] = -ty_s

            # North neighbor
            if j < ny - 1:
                col = row + nx
                A[row, col] = -ty_n

    # ---------- Wells → add to RHS ----------
    for well in model.wells:
        q = well.get_rhs_contribution(nx, ny)
        b += q

    A = A.tocsr()
    return A, b


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def _constant_head_value(model, i, j):
    """
    Search boundaries list for a constant-head boundary at (i,j).
    Return the constant head value, or None if not constant-head.
    """
    for bc in model.boundaries:
        if bc["type"] == "CONSTANT_HEAD" and bc["i"] == i and bc["j"] == j:
            return bc["value"]
    return None


def _trans_x(model, thickness, i, j, inbr, jnbr):
    """
    Transmissivity between cell (i,j) and west/east neighbor.
    Harmonic average recommended in MODFLOW & USBR (§5-14).
    """
    nx, ny = model.grid.nx, model.grid.ny
    if inbr < 0 or inbr >= nx:
        return 0.0

    K1 = model.properties.Kx[0, i, j]
    K2 = model.properties.Kx[0, inbr, j]

    t1 = thickness[i, j]
    t2 = thickness[inbr, j]

    if t1 <= 0 or t2 <= 0:
        return 0.0

    # harmonic average transmissivity
    return 2.0 / (1.0/(K1 * t1) + 1.0/(K2 * t2))


def _trans_y(model, thickness, i, j, inbr, jnbr):
    """
    Transmissivity between cell (i,j) and south/north neighbor.
    Harmonic average (Todd 2005; USBR Ground Water Manual).
    """
    nx, ny = model.grid.nx, model.grid.ny
    if jnbr < 0 or jnbr >= ny:
        return 0.0

    K1 = model.properties.Ky[0, i, j]
    K2 = model.properties.Ky[0, i, jnbr]

    t1 = thickness[i, j]
    t2 = thickness[i, jnbr]

    if t1 <= 0 or t2 <= 0:
        return 0.0

    return 2.0 / (1.0/(K1 * t1) + 1.0/(K2 * t2))
