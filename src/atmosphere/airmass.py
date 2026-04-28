"""
Air mass calculations for astronomical observations.

This module provides functions to compute the air mass (optical path
length through the atmosphere) using the Kasten & Young (1989) formula,
and the relative atmospheric pressure as a function of elevation.

Air mass is the ratio of the optical path length through the atmosphere
to the path length at the zenith. It is a key parameter for atmospheric
extinction and scattering models.

All angles are in degrees unless explicitly stated otherwise.
"""

import numpy as np


def air_mass(altitude_deg: float) -> float:
    """Compute the relative air mass for a given altitude above the horizon.

    Uses the Kasten & Young (1989) formula, valid for altitudes above
    the horizon (h > 0°).

        X = 1 / (sin(h_rad) + 0.50572 * (h_deg + 6.07995)^(-1.6364))

    For altitudes at or below the horizon, returns a large value
    (effectively infinite) to indicate that the air mass is not defined.

    :param altitude_deg: Altitude of the object above the horizon in degrees
    :return: Air mass (dimensionless, >= 1.0)
    """
    if altitude_deg <= 0.0:
        # At or below the horizon — effectively infinite optical path
        return float('inf')

    h_rad = np.radians(altitude_deg)
    h_deg = altitude_deg

    # Kasten & Young (1989)
    denom = (np.sin(h_rad)
             + 0.50572 * (h_deg + 6.07995) ** (-1.6364))
    X = 1.0 / denom

    # Clamp to reasonable maximum
    if X > 100.0:
        return float('inf')

    return float(X)


def relative_pressure(elevation_m: float) -> float:
    """Compute the relative atmospheric pressure at a given elevation.

    Uses a simple exponential approximation:
        P / P0 = exp(-elevation / 8000.0)

    where 8000 m is the approximate scale height of the atmosphere.

    :param elevation_m: Elevation above sea level in meters
    :return: Relative pressure (P/P0), dimensionless (0.0–1.0)
    """
    return float(np.exp(-elevation_m / 8000.0))
