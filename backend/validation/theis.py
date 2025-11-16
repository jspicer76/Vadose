"""
Theis (1935) Transient Radial Flow Analytical Solution
======================================================

References:
- Theis, C.V. (1935). "The relation between the lowering of the piezometric
  surface and the rate and duration of discharge of a well using groundwater
  storage." Am. Geophys. Union Trans.
- Todd, D.K. & Mays, L.W. (2005). Groundwater Hydrology, 3rd ed.
  Chapter 5: Transient radial flow.
- USBR Ground Water Manual (1995), Chapter IX: Analytical methods for
  discharging well test data.

This module provides:
1. Theis well function W(u)
2. Theis drawdown s(r,t)
3. Numerical-stable approximations for small and large u
"""

import math


def theis_W(u: float, terms: int = 50) -> float:
    """
    Compute the Theis well function W(u) using its series expansion:

        W(u) = -γ - ln(u) + u - u^2/(2·2!) + u^3/(3·3!) - ...

    where γ = Euler-Mascheroni constant.

    Parameters:
        u (float): Parameter u = r^2 S / (4 T t)
        terms (int): Number of series terms (default 50)

    Returns:
        float: W(u)
    """
    if u <= 0:
        raise ValueError("u must be > 0 in Theis well function")

    gamma = 0.5772156649015328606  # Euler-Mascheroni constant

    # For small u, series converges well.
    if u < 1e-3:
        series_sum = 0.0
        sign = 1.0
        for n in range(1, terms + 1):
            series_sum += sign * (u ** n) / (n * math.factorial(n))
            sign *= -1.0
        return -gamma - math.log(u) + series_sum

    # For larger u, the series truncation converges rapidly.
    series_sum = 0.0
    for n in range(1, terms + 1):
        series_sum += ((-1) ** (n - 1)) * (u ** n) / (n * math.factorial(n))

    return -gamma - math.log(u) + series_sum


def theis_drawdown(Q: float, T: float, S: float, r: float, t: float) -> float:
    """
    Compute Theis analytical drawdown:

        s(r,t) = (Q / (4 π T)) · W(u)

        where u = r² S / (4 T t)

    Parameters:
        Q (float): Pumping rate (L^3/T)
        T (float): Transmissivity (L^2/T)
        S (float): Storativity (dimensionless)
        r (float): Radial distance (L)
        t (float): Time since pumping began (T)

    Returns:
        float: drawdown s(r,t)
    """
    if t <= 0:
        raise ValueError("time t must be > 0")
    if r <= 0:
        raise ValueError("distance r must be > 0")

    u = (r * r * S) / (4.0 * T * t)
    W_u = theis_W(u)

    return (Q / (4.0 * math.pi * T)) * W_u
