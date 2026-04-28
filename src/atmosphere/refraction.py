"""
Atmospheric refraction calculations.

This module implements atmospheric refraction corrections for celestial
objects using the Saemundsson (1986) formula, with temperature and
pressure corrections applied.

All angles are in degrees unless explicitly stated otherwise.
"""

import numpy as np


def refraction_correction(apparent_altitude_deg: float,
                          temp_c: float = 10.0,
                          pressure_mbar: float = 1013.0) -> float:
    """Compute the atmospheric refraction correction for a given altitude.

    Uses the Saemundsson (1986) formula. The correction is the amount by
    which an object's apparent altitude exceeds its true altitude.

    R (arcmin) = 1.02 / tan(h + 10.3 / (h + 5.11))

    where h is the apparent altitude in degrees.

    The correction is then adjusted for actual temperature and pressure:
        corrected_R = R * (P / 1013.0) * (283.0 / (T + 273.15))

    :param apparent_altitude_deg: Apparent altitude of the object in degrees
    :param temp_c: Temperature in degrees Celsius (default 10.0)
    :param pressure_mbar: Atmospheric pressure in millibars (default 1013.0)
    :return: Refraction correction in degrees (positive for above-horizon)
    """
    h = apparent_altitude_deg

    # Saemundsson formula only valid near/above horizon
    if h < -5.0:
        # Very low below horizon — negligible refraction
        return 0.0

    # Handle near-horizon and below-horizon gracefully
    h_clamped = max(h, -5.0)

    # R in arcminutes
    R_arcmin = 1.02 / np.tan(np.radians(h_clamped + 10.3 / (h_clamped + 5.11)))

    # Temperature/pressure correction
    T_kelvin = temp_c + 273.15
    R_corrected = R_arcmin * (pressure_mbar / 1013.0) * (283.0 / T_kelvin)

    # Convert arcminutes to degrees
    return float(R_corrected / 60.0)


def apparent_from_true(true_altitude_deg: float,
                       temp_c: float = 10.0,
                       pressure_mbar: float = 1013.0) -> float:
    """Compute the apparent altitude from the true (geometric) altitude.

    This is an iterative inverse of the Saemundsson formula, since the
    formula takes apparent altitude as input but we need to go from
    true → apparent.

    Uses Newton's method or direct iteration (typically converges in
    2-3 iterations).

    :param true_altitude_deg: True (geometric) altitude in degrees
    :param temp_c: Temperature in degrees Celsius (default 10.0)
    :param pressure_mbar: Atmospheric pressure in millibars (default 1013.0)
    :return: Apparent altitude in degrees
    """
    # For very low altitudes, just start with true + a small guess
    h_true = true_altitude_deg

    if h_true < -5.0:
        return h_true  # negligible refraction

    # Initial guess: apparent ~ true + 0.5° near horizon, less at zenith
    h_app = h_true + 0.5 / 60.0  # Start with ~0.5 arcmin correction

    # Newton's method: solve f(h_app) = h_app - R(h_app) - h_true = 0
    for _ in range(10):
        R_deg = refraction_correction(h_app, temp_c, pressure_mbar)
        residual = (h_app - R_deg) - h_true

        if abs(residual) < 1e-8:
            break

        # Approximate derivative using finite difference
        eps = 1e-6
        R_deg_eps = refraction_correction(h_app + eps, temp_c, pressure_mbar)
        dR_dh = (R_deg_eps - R_deg) / eps
        f_prime = 1.0 - dR_dh

        if abs(f_prime) > 1e-12:
            h_app -= residual / f_prime
        else:
            # Fallback: simple step
            h_app += residual * 0.5

    return float(h_app)
