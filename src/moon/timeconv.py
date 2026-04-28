"""
Astronomical time conversion utilities.

This module provides functions for converting between calendar dates and
astronomical time scales, including Julian Day Numbers, sidereal time
(Greenwich and Local), and the Earth rotation correction ΔT.

All functions use numpy for floating-point precision and trigonometric
operations. Angles are in degrees unless explicitly stated otherwise.
"""

import numpy as np


def julian_date(year: float, month: float, day: float,
                hour: float = 0.0, minute: float = 0.0,
                second: float = 0.0) -> float:
    """Compute the Julian Day Number for a given Gregorian calendar date.

    Uses the standard astronomical algorithm (valid for Gregorian calendar
    dates after 1582-10-15). The result includes the fractional day from
    the provided hour, minute, and second.

    :param year: Year (e.g. 2026)
    :param month: Month number (1 = January, 12 = December)
    :param day: Day of month (1-based)
    :param hour: Hour (0-23, default 0.0)
    :param minute: Minute (0-59, default 0.0)
    :param second: Second (0-59, default 0.0)
    :return: Julian Day Number as a float
    """
    # Fractional day from time
    fractional_day = (hour + minute / 60.0 + second / 3600.0) / 24.0

    # Algorithm: shift month/year so the year starts in March
    a = np.floor((14 - month) / 12.0)
    y = year + 4800.0 - a
    m = month + 12.0 * a - 3.0

    # Julian Day Number (Gregorian)
    jd = (day + fractional_day
          + np.floor((153.0 * m + 2.0) / 5.0)
          + 365.0 * y
          + np.floor(y / 4.0)
          - np.floor(y / 100.0)
          + np.floor(y / 400.0)
          - 32045.0)

    return float(jd)


def gmst(jd: float) -> float:
    """Compute Greenwich Mean Sidereal Time for a given Julian Day.

    GMST is the hour angle of the vernal equinox at the Greenwich meridian.
    The result is in hours, wrapped to the range [0, 24).

    Formula: GMST = 18.697374558 + 24.06570982441908 * (jd - 2451545.0)

    :param jd: Julian Day Number
    :return: Greenwich Mean Sidereal Time in hours (0-24)
    """
    gmst_hours = 18.697374558 + 24.06570982441908 * (jd - 2451545.0)
    # Wrap to [0, 24)
    return float(gmst_hours % 24.0)


def lmst(gmst_hours: float, longitude_deg: float) -> float:
    """Convert Greenwich Mean Sidereal Time to Local Sidereal Time.

    LST adjusts GMST by the observer's east longitude. The result is
    wrapped to the range [0, 24) hours.

    :param gmst_hours: Greenwich Mean Sidereal Time in hours (0-24)
    :param longitude_deg: Observer's east longitude in degrees
                          (west longitudes are negative)
    :return: Local Sidereal Time in hours (0-24)
    """
    # 15 degrees of longitude = 1 hour of sidereal time
    lst_hours = gmst_hours + longitude_deg / 15.0
    return float(lst_hours % 24.0)


def delta_t(year: float) -> float:
    """Approximate ΔT (Delta T) in seconds for a given year.

    ΔT is the difference between Terrestrial Time (TT) and Universal Time
    (UT1): ΔT = TT - UT1. This function uses a simplified version of the
    Morrison & Stephenson (2004) approximation.

    :param year: Year (e.g. 2026.5 for mid-2026)
    :return: ΔT in seconds
    """
    return 32.184 + (year - 2000.0) * 0.33
