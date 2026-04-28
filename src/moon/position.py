"""
Moon and sun position calculations.

This module provides functions to compute the approximate geocentric
positions of the Sun and Moon using simplified analytical methods from
the Astronomical Almanac. It also provides coordinate conversion
functions (equatorial to horizontal) and apparent position corrections.

All angles are in degrees unless explicitly stated otherwise.
All trigonometric operations use numpy.
"""

import numpy as np
from moon import timeconv as tc


def _ecliptic_obliquity(jd: float) -> float:
    """Compute the obliquity of the ecliptic (mean) for a given JD.

    Uses the standard IAU 1976 formula.

    :param jd: Julian Day Number
    :return: Obliquity of the ecliptic in degrees
    """
    T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000.0
    eps = (23.439291
           - 0.0130042 * T
           - 1.638e-7 * T ** 2
           + 5.036e-7 * T ** 3)
    return eps


def _ecliptic_to_equatorial(lon_deg: float, lat_deg: float,
                            eps_deg: float) -> tuple[float, float]:
    """Convert ecliptic coordinates to equatorial (RA, Dec).

    :param lon_deg: Ecliptic longitude in degrees
    :param lat_deg: Ecliptic latitude in degrees
    :param eps_deg: Obliquity of the ecliptic in degrees
    :return: Tuple of (right_ascension_deg, declination_deg)
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    eps = np.radians(eps_deg)

    x = np.cos(lon) * np.cos(lat)
    y = np.cos(eps) * np.sin(lon) * np.cos(lat) - np.sin(eps) * np.sin(lat)
    z = np.sin(eps) * np.sin(lon) * np.cos(lat) + np.cos(eps) * np.sin(lat)

    ra = np.degrees(np.arctan2(y, x))
    dec = np.degrees(np.arcsin(z))

    # RA in [0, 360)
    ra = ra % 360.0

    return float(ra), float(dec)


def _equatorial_to_horizontal(ra_deg: float, dec_deg: float,
                              lat_deg: float, lon_deg: float,
                              jd: float) -> tuple[float, float]:
    """Convert equatorial coordinates (RA, Dec) to horizontal (alt, az).

    :param ra_deg: Right ascension in degrees
    :param dec_deg: Declination in degrees
    :param lat_deg: Observer's latitude in degrees (north positive)
    :param lon_deg: Observer's longitude in degrees (east positive)
    :param jd: Julian Day Number
    :return: Tuple of (altitude_deg, azimuth_deg)
             altitude: degrees above horizon (-90 to +90)
             azimuth: degrees from north through east (0-360)
    """
    # Compute Local Sidereal Time
    g = tc.gmst(jd)
    lst_deg = tc.lmst(g, lon_deg) * 15.0  # convert hours to degrees

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)
    ha = np.radians(lst_deg - ra_deg)  # hour angle

    # Altitude
    alt = np.arcsin(
        np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    )

    # Azimuth (measured from north through east)
    az = np.arctan2(
        -np.sin(ha),
        np.tan(dec) * np.cos(lat) - np.sin(lat) * np.cos(ha)
    )

    alt_deg = float(np.degrees(alt))
    az_deg = float(np.degrees(az) % 360.0)

    return alt_deg, az_deg


def sun_position(jd: float) -> tuple[float, float]:
    """Compute the approximate geocentric position of the Sun.

    Uses a simplified model based on the mean ecliptic longitude of the
    Sun, with Keplerian corrections for the equation of center.

    :param jd: Julian Day Number
    :return: Tuple (ra_deg, dec_deg) — right ascension and declination in degrees
    """
    T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000.0

    # Mean longitude of the Sun (degrees)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2

    # Mean anomaly of the Sun (degrees)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2

    # Equation of center
    C = ((1.914602 - 0.004817 * T - 0.000014 * T ** 2) * np.sin(np.radians(M))
         + (0.019993 - 0.000101 * T) * np.sin(np.radians(2 * M))
         + 0.000289 * np.sin(np.radians(3 * M)))

    # Ecliptic longitude of the Sun
    sun_lon = L0 + C

    # The Sun's ecliptic latitude is effectively 0
    sun_lat = 0.0

    # Obliquity of the ecliptic
    eps = _ecliptic_obliquity(jd)

    # Convert to equatorial
    ra, dec = _ecliptic_to_equatorial(sun_lon, sun_lat, eps)

    return ra, dec


def moon_position(jd: float) -> tuple[float, float, float]:
    """Compute the approximate geocentric position of the Moon.

    Uses a simplified analytical method based on the Astronomical Almanac
    (Brown's lunar theory, truncated). Computes mean longitude, mean
    elongation, mean anomaly, and argument of latitude, then applies
    perturbation terms.

    :param jd: Julian Day Number
    :return: Tuple (ra_deg, dec_deg, distance_km)
             — right ascension, declination, and distance in km
    """
    T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000.0
    D2R = np.pi / 180.0

    # Moon's mean longitude (degrees)
    Lp = (218.3165 + 481267.8813 * T
          - 0.00161 * T ** 2 + 0.000005 * T ** 3)

    # Sun's mean anomaly (degrees)
    M = (357.5291 + 35999.0503 * T - 0.000153 * T ** 2
         + 0.000004 * T ** 3)

    # Moon's mean anomaly (degrees)
    Mp = (134.9634 + 477198.8676 * T
          + 0.00899 * T ** 2 + 0.000017 * T ** 3)

    # Moon's argument of latitude (mean distance from ascending node, degrees)
    F = (93.2720 + 483202.0175 * T
         - 0.00355 * T ** 2 - 0.000015 * T ** 3)

    # Mean elongation of the Moon (Moon's mean longitude - Sun's mean longitude)
    D = (297.8502 + 445267.1114 * T
         - 0.00163 * T ** 2 + 0.000005 * T ** 3)

    # --- Perturbations in ecliptic longitude ---
    sum_lon = (
        + 6288.0 * np.sin(Mp * D2R)
        + 1274.0 * np.sin((2.0 * D - Mp) * D2R)
        + 658.0 * np.sin(2.0 * D * D2R)
        + 214.0 * np.sin(2.0 * Mp * D2R)
        - 186.0 * np.sin(M * D2R)
        + 114.0 * np.sin(2.0 * F * D2R)
        + 59.0 * np.sin((2.0 * D - 2.0 * Mp) * D2R)
        + 53.0 * np.sin((2.0 * D - M - Mp) * D2R)
        + 44.0 * np.sin((M + Mp) * D2R)
        - 31.0 * np.sin((-M + 2.0 * D - Mp) * D2R)
        + 26.0 * np.sin((2.0 * D + Mp) * D2R)
        + 19.0 * np.sin((M - Mp) * D2R)
        + 18.0 * np.sin((2.0 * D - M) * D2R)
        + 17.0 * np.sin(M * D2R) * 0.0  # placeholder (combined above)
        + 14.0 * np.sin((2.0 * D - 2.0 * Mp + 2.0 * F) * D2R)  # evection
    )

    # --- Perturbations in ecliptic latitude ---
    sum_lat = (
        + 5128.0 * np.sin(F * D2R)
        + 2806.0 * np.sin((Mp + F) * D2R)
        + 2777.0 * np.sin((Mp - F) * D2R)
        + 1736.0 * np.sin((2.0 * D - Mp + F) * D2R)
        + 554.0 * np.sin((2.0 * D - Mp - F) * D2R)
        + 463.0 * np.sin((2.0 * D - F) * D2R)
        + 326.0 * np.sin((2.0 * D + F) * D2R)
        + 172.0 * np.sin((2.0 * Mp + F) * D2R)
        + 92.0 * np.sin((2.0 * Mp - F) * D2R)
        + 87.0 * np.sin((2.0 * D + Mp - F) * D2R)
    )

    # --- Perturbations in distance (km) ---
    sum_dist = (
        - 20905.0 * np.cos(Mp * D2R)
        - 3699.0 * np.cos((2.0 * D - Mp) * D2R)
        - 2956.0 * np.cos(2.0 * D * D2R)
        - 570.0 * np.cos(2.0 * Mp * D2R)
        + 246.0 * np.cos(2.0 * D - 2.0 * Mp * D2R)
        - 205.0 * np.cos((2.0 * D - M - Mp) * D2R)
        - 171.0 * np.cos((M - Mp) * D2R)
        - 152.0 * np.cos((M + Mp) * D2R)
        - 122.0 * np.cos((M + 2.0 * D - Mp) * D2R)  # approx
    )

    # Mean distance in km
    mean_dist_km = 385000.56

    # Ecliptic longitude (degrees)
    # The perturbation sum_lon is in 0.001 degrees (arcminutes * 60)
    ecl_lon = Lp + sum_lon / 3600.0  # convert arcseconds to degrees

    # Ecliptic latitude (degrees)
    ecl_lat = sum_lat / 3600.0  # convert arcseconds to degrees

    # Distance (km)
    distance_km = mean_dist_km + sum_dist  # sum_dist already in km

    # Obliquity of the ecliptic
    eps = _ecliptic_obliquity(jd)

    # Convert to equatorial
    ra, dec = _ecliptic_to_equatorial(ecl_lon, ecl_lat, eps)

    return ra, dec, distance_km


def moon_alt_az(lat_deg: float, lon_deg: float, jd: float
                ) -> tuple[float, float]:
    """Compute the geocentric altitude and azimuth of the Moon.

    :param lat_deg: Observer's latitude in degrees (north positive)
    :param lon_deg: Observer's longitude in degrees (east positive)
    :param jd: Julian Day Number
    :return: Tuple (altitude_deg, azimuth_deg)
             altitude: degrees above horizon
             azimuth: degrees from north through east (0-360)
    """
    ra, dec, _ = moon_position(jd)
    return _equatorial_to_horizontal(ra, dec, lat_deg, lon_deg, jd)


def moon_apparent_alt_az(lat_deg: float, lon_deg: float, jd: float,
                         temp_c: float = 10.0,
                         pressure_mbar: float = 1013.0
                         ) -> tuple[float, float]:
    """Compute the apparent altitude and azimuth of the Moon, including
    atmospheric refraction correction.

    :param lat_deg: Observer's latitude in degrees (north positive)
    :param lon_deg: Observer's longitude in degrees (east positive)
    :param jd: Julian Day Number
    :param temp_c: Temperature in degrees Celsius (default 10.0)
    :param pressure_mbar: Atmospheric pressure in millibars (default 1013.0)
    :return: Tuple (altitude_deg, azimuth_deg)
             apparent altitude (refracted) and azimuth
    """
    from atmosphere import refraction

    ra, dec, _ = moon_position(jd)
    alt_deg, az_deg = _equatorial_to_horizontal(ra, dec, lat_deg, lon_deg, jd)

    # Apply refraction to the true altitude to get apparent altitude
    apparent_alt = refraction.apparent_from_true(alt_deg, temp_c, pressure_mbar)

    return apparent_alt, az_deg
