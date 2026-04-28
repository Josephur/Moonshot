"""
Atmospheric scattering calculations for moon color.

This module computes the color tint of the Moon as seen from a given
location, accounting for Rayleigh scattering (wavelength-dependent)
and Mie scattering (humidity-dependent) along the line of sight.

The output is a set of RGB multipliers that can be applied to a base
moon texture to simulate atmospheric reddening at low altitudes.

Model:
  - Rayleigh scattering ~ λ⁻⁴ (blue is scattered ~5x more than red)
  - Mie scattering is roughly wavelength-independent (aerosols/haze)
  - Higher humidity → more Mie scattering → dimmer, whiter moon
  - Low altitude → long optical path → more Rayleigh → redder moon
  - Altitude > 30° → near-white (~0.95+ multipliers)

All angles are in degrees unless explicitly stated otherwise.
"""

import numpy as np
from atmosphere import airmass


# Reference wavelengths for RGB channels (nanometers)
# Representative wavelengths for photopic vision
_RED_WL: float = 700.0
_GREEN_WL: float = 550.0
_BLUE_WL: float = 450.0


def _rayleigh_scattering_factor(wavelength_nm: float) -> float:
    """Compute the relative Rayleigh scattering coefficient.

    Rayleigh scattering is proportional to λ⁻⁴, normalized so that
    the green channel (550 nm) has coefficient 1.0.

    :param wavelength_nm: Wavelength in nanometers
    :return: Relative scattering coefficient (dimensionless)
    """
    return (550.0 / wavelength_nm) ** 4.0


def _mie_scattering_factor(humidity_pct: float) -> float:
    """Compute the Mie scattering optical depth from humidity.

    Mie scattering from aerosols (water droplets, dust) increases with
    humidity. This model returns a scale factor relative to a dry,
    clean atmosphere.

    At 0% humidity:  factor ≈ 0.3 (low aerosol, clear air)
    At 50% humidity: factor ≈ 0.5 (moderate haze)
    At 100% humidity: factor ≈ 2.0 (heavy haze/fog, high optical depth)

    :param humidity_pct: Relative humidity in percent (0-100)
    :return: Mie scattering scale factor (dimensionless)
    """
    h = float(np.clip(humidity_pct, 0.0, 100.0))
    # Empirical curve: gentle rise at low humidity, steeper at high
    return 0.3 + (h / 100.0) ** 3 * 1.7


def moon_color_tint(altitude_deg: float,
                    temp_c: float = 10.0,
                    pressure_mbar: float = 1013.0,
                    humidity_pct: float = 50.0) -> tuple[float, float, float]:
    """Compute RGB color multipliers for the Moon at a given altitude.

    The tint accounts for:
    1. Optical path length (air mass × relative pressure)
    2. Rayleigh scattering (λ⁻⁴ — blue scatters more, making moon redder)
    3. Mie scattering from humidity (roughly neutral across wavelengths,
       making the moon appear dimmer and less color-saturated)

    At high altitudes (> 30°) the multipliers are near 1.0 (white).
    At low altitudes, Rayleigh scattering removes blue light, giving
    a reddish tint.  High humidity adds neutral attenuation and reduces
    the color contrast.

    :param altitude_deg: Apparent altitude of the Moon in degrees
    :param temp_c: Temperature in degrees Celsius (default 10.0)
    :param pressure_mbar: Atmospheric pressure in millibars (default 1013.0)
    :param humidity_pct: Relative humidity in percent 0-100 (default 50.0)
    :return: Tuple (r, g, b) — RGB multipliers in range [0.0, 1.0]
    """
    am = airmass.air_mass(altitude_deg)

    if am == float('inf') or altitude_deg <= 0.0:
        return 0.0, 0.0, 0.0

    # Optical path: air mass scaled by relative pressure
    rel_p = pressure_mbar / 1013.0
    optical_path = am * rel_p

    # Rayleigh scattering coefficients (normalized to green = 1.0)
    rayleigh_r = _rayleigh_scattering_factor(_RED_WL)     # ~0.40
    rayleigh_g = _rayleigh_scattering_factor(_GREEN_WL)   #  1.00
    rayleigh_b = _rayleigh_scattering_factor(_BLUE_WL)    # ~2.26

    # Base Rayleigh scattering optical depth scale
    # Tuned so that at zenith (optical_path=1) the blue min channel ≈ 0.95
    rayleigh_base = 0.04

    # Mie scattering optical depth from humidity (neutral across channels)
    mie_od = _mie_scattering_factor(humidity_pct) * 0.02

    # Total optical depth per channel = Rayleigh + Mie
    tau_r = rayleigh_r * rayleigh_base * optical_path + mie_od * optical_path
    tau_g = rayleigh_g * rayleigh_base * optical_path + mie_od * optical_path
    tau_b = rayleigh_b * rayleigh_base * optical_path + mie_od * optical_path

    # Transmission via Beer-Lambert law
    trans_r = np.exp(-tau_r)
    trans_g = np.exp(-tau_g)
    trans_b = np.exp(-tau_b)

    # Normalize so the brightest channel (red) reaches ~0.98 at zenith
    max_channel = max(trans_r, trans_g, trans_b)
    if max_channel > 0.0 and max_channel < 0.98:
        scale = 0.98 / max_channel
        trans_r *= scale
        trans_g *= scale
        trans_b *= scale

    # Clamp to [0, 1]
    r = float(np.clip(trans_r, 0.0, 1.0))
    g = float(np.clip(trans_g, 0.0, 1.0))
    b = float(np.clip(trans_b, 0.0, 1.0))

    return r, g, b
