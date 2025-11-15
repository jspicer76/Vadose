import numpy as np
from scipy.sparse import lil_matrix

# ==============================================================================
#  EXPAND LOCATION-BASED BOUNDARIES INTO CELL-BASED BOUNDARIES
# ==============================================================================

def expand_location_boundaries(model):
    """
    Converts location-based constant-head boundaries like:
        {"type": "CONSTANT_HEAD", "value": 100, "location": "LEFT"}
    into explicit cell-index boundaries:
        {"type": "CONSTANT_HEAD", "value": 100, "i": 0, "j": j}

    This allows simple model definitions while still giving the
    solver explicit (i, j) coordinates for boundary conditions.

    Returns:
        List of BC dictionaries with explicit (i, j) coordinates.
    """

    nx, ny = model.grid.nx, model.grid.ny
    expanded = []

    for bc in model.boundaries:

        # Keep non-constant-head boundaries unchanged
        if bc["type"] != "CONSTANT_HEAD":
            expanded.append(bc)
            continue

        # If (i, j) are already given → keep as is
        if "i" in bc and "j" in bc:
            expanded.append(bc)
            continue

        # Interpret location keywords
        location = bc.get("location", "").upper()
        value = bc["value"]

        if location == "LEFT":
            for j in range(ny):
                expanded.append({"type": "CONSTANT_HEAD", "value": value, "i": 0, "j": j})

        elif location == "RIGHT":
            for j in range(ny):
                expanded.append({"type": "CONSTANT_HEAD", "value": value, "i": nx - 1, "j": j})

        elif location == "TOP":
            for i in range(nx):
                expanded.append({"type": "CONSTANT_HEAD", "value": value, "i": i, "j": ny - 1})

        elif location == "BOTTOM":
            for i in range(nx):
                expanded.append({"type": "CONSTANT_HEAD", "value": value, "i": i, "j": 0})

        else:
            # Unknown location → skip silently or raise
            continue

    return expanded


# ==============================================================================
#  MATRIX ASSEMBLY
# ==============================================================================

def assemble_matrix(model, head_prev):
    """
    Assemble the finite-difference matrix (A) and RHS vector (b)
    for confined or unconfined saturated groundwater flow.

    Implements BCF-style unconfined correction:
        saturated_thickness = max(head - bottom, min_thick)

    Returns:
        A (CSR sparse matrix)
        b (RHS vector)
        grid_shape = (nx, ny)
    """

    grid = model.grid
    props = model.properties

    nx, ny = grid.nx, grid.ny
    nlay = props.nlay

    assert nlay == 1, "Current implementation handles 1 layer only."

    # Expand boundaries (location → per-cell)
    boundaries = expand_location_boundaries(model)

    # Flattened system size
    N = nx * ny

    # Sparse matrix and RHS vector
    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N)

    # ==============================================================================
    #  SATURATED THICKNESS (for unconfined flow)
    # ==============================================================================
    thickness = np.zeros((nx, ny))
    base_thickness = props.thickness[0]

    if props.confined:
        thickness[:, :] = base_thickness
    else:
        # Unconfined BCF method
        bottom = np.zeros((nx, ny))           # Placeholder: no bottom elevations yet
        min_thick = 0.1

        for i in range(nx):
            for j in range(ny):
                sat = max(head_prev[i, j] - bottom[i, j], min_thick)
                thickness[i, j] = sat

    # ==============================================================================
    #  MAIN LOOP: Build A and b
    # ==============================================================================
    for j in range(ny):
        for i in range(nx):

            row = j * nx + i  # flatten 2D → 1D index

            # ---------------------------------------------------------------
            # CONSTANT HEAD BC
            # ---------------------------------------------------------------
            ch = _constant_head_value(boundaries, i, j)
            if ch is not None:
                A[row, row] = 1.0
                b[row] = ch
                continue

            # ---------------------------------------------------------------
            # TRANSMISSIVITY VALUES
            # ---------------------------------------------------------------
            Kx = props.Kx[0, i, j]
            Ky = props.Ky[0, i, j]

            # Own transmissivity
            Tx = Kx * thickness[i, j]
            Ty = Ky * thickness[i, j]

            # Neighbor transmissivities (harmonic)
            tx_w = _trans_x(model, thickness, i, j, i - 1, j)
            tx_e = _trans_x(model, thickness, i, j, i + 1, j)
            ty_s = _trans_y(model, thickness, i, j, i, j - 1)
            ty_n = _trans_y(model, thickness, i, j, i, j + 1)

            # Diagonal coefficient
            A[row, row] = tx_w + tx_e + ty_s + ty_n

            # West neighbor
            if i > 0:
                A[row, row - 1] = -tx_w

            # East neighbor
            if i < nx - 1:
                A[row, row + 1] = -tx_e

            # South neighbor
            if j > 0:
                A[row, row - nx] = -ty_s

            # North neighbor
            if j < ny - 1:
                A[row, row + nx] = -ty_n

    # ==============================================================================
    #  ADD WELLS
    # ==============================================================================
    for well in model.wells:
        q = well.get_rhs_contribution(nx, ny)
        b += q

    return A.tocsr(), b, (nx, ny)


# ==============================================================================
#  BOUNDARY HELPERS
# ==============================================================================

def _constant_head_value(boundaries, i, j):
    """Return constant-head value at (i,j) or None."""
    for bc in boundaries:
        if bc["type"] == "CONSTANT_HEAD" and bc["i"] == i and bc["j"] == j:
            return bc["value"]
    return None


# ==============================================================================
#  TRANSMISSIVITY HELPERS
# ==============================================================================

def _trans_x(model, thickness, i, j, inbr, jnbr):
    """
    Transmissivity between (i,j) and west/east neighbor.
    Harmonic averaging recommended (USBR §5-14).
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

    return 2.0 / (1.0/(K1 * t1) + 1.0/(K2 * t2))


def _trans_y(model, thickness, i, j, inbr, jnbr):
    """
    Transmissivity between (i,j) and south/north neighbor.
    Harmonic averaging (Todd 2005; USBR Ground Water Manual).
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
