"""
Moon phase calculator.

This module computes the illuminated fraction of the Moon, the phase
name, the position angle of the bright limb (terminator angle), and
whether the Moon is waxing or waning — all based on the geocentric
positions of the Sun and Moon.

All angles are in degrees unless explicitly stated otherwise.
"""

import numpy as np
from moon import position as moonpos


def _sun_moon_separation(sun_ra_deg: float, sun_dec_deg: float,
                         moon_ra_deg: float, moon_dec_deg: float) -> float:
    """Compute the angular separation between the Sun and Moon.

    Uses the spherical law of cosines.

    :param sun_ra_deg: Sun right ascension in degrees
    :param sun_dec_deg: Sun declination in degrees
    :param moon_ra_deg: Moon right ascension in degrees
    :param moon_dec_deg: Moon declination in degrees
    :return: Angular separation in degrees
    """
    d_ra = np.radians(moon_ra_deg - sun_ra_deg)
    sun_dec = np.radians(sun_dec_deg)
    moon_dec = np.radians(moon_dec_deg)

    sep = np.arccos(
        np.sin(sun_dec) * np.sin(moon_dec)
        + np.cos(sun_dec) * np.cos(moon_dec) * np.cos(d_ra)
    )
    return float(np.degrees(sep))


def is_waxing(sun_ra_deg: float, moon_ra_deg: float) -> bool:
    """Determine if the Moon is waxing (growing toward full).

    The Moon is waxing when its right ascension is greater than the
    Sun's (i.e., it is east of the Sun), which means more of its
    illuminated side is visible from Earth.

    :param sun_ra_deg: Sun right ascension in degrees
    :param moon_ra_deg: Moon right ascension in degrees
    :return: True if the Moon is waxing, False if waning
    """
    # Normalize RA differences to [-180, 180]
    diff = (moon_ra_deg - sun_ra_deg) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff > 0.0


def illumination(jd: float, sun_pos: tuple[float, float] | None = None,
                 moon_pos: tuple[float, float, float] | None = None
                 ) -> float:
    """Compute the illuminated fraction of the Moon.

    Returns a value between 0.0 (new moon) and 1.0 (full moon) based on
    the geocentric elongation angle between the Sun and Moon.

    If sun_pos or moon_pos are not provided, they are computed
    automatically from the Julian Day.

    :param jd: Julian Day Number
    :param sun_pos: Optional pre-computed (ra_deg, dec_deg) for the Sun
    :param moon_pos: Optional pre-computed (ra_deg, dec_deg, distance_km)
                     for the Moon
    :return: Illuminated fraction (0.0 = new, 1.0 = full)
    """
    if sun_pos is None:
        sun_ra, sun_dec = moonpos.sun_position(jd)
    else:
        sun_ra, sun_dec = sun_pos

    if moon_pos is None:
        moon_ra, moon_dec, _ = moonpos.moon_position(jd)
    else:
        moon_ra, moon_dec, _ = moon_pos

    # Angular separation between Sun and Moon
    sep_deg = _sun_moon_separation(sun_ra, sun_dec, moon_ra, moon_dec)

    # Illuminated fraction:
    # k = (1 - cos(separation)) / 2
    # where separation = 180° at full moon, 0° at new moon
    # More precisely, elongation angle from the Sun:
    #   k = 0 at conjunction (New Moon), k = 1 at opposition (Full Moon)
    elongation = np.radians(sep_deg)
    k = (1.0 - np.cos(elongation)) / 2.0

    return float(k)


def phase_name(illumination_fraction: float, waxing: bool) -> str:
    """Return the phase name for a given illumination and waxing state.

    The eight traditional lunar phases:

    Illumination k is defined as:
        k = 0.0 at New Moon (elongation 0° — conjunction)
        k = 0.5 at Quarter (elongation 90°)
        k = 1.0 at Full Moon (elongation 180° — opposition)

    Mapping:
        ============= =================
        k range        Phase Name
        ============= =================
        < 0.03        New Moon
        0.03–0.40     Waxing/Waning Crescent
        0.40–0.60     First/Last Quarter
        0.60–0.97     Waxing/Waning Gibbous
        > 0.97        Full Moon
        ============= =================
    """
    # Round to handle floating-point edge cases
    k = float(illumination_fraction)

    if k < 0.03:
        return "New Moon"

    if k >= 0.97:
        return "Full Moon"

    # Near quarter (elongation ~90°, k ~ 0.5)
    if 0.40 <= k <= 0.60:
        if waxing:
            return "First Quarter"
        else:
            return "Last Quarter"

    # Between New and Quarter (elongation 0-90°)
    if k < 0.40:
        if waxing:
            return "Waxing Crescent"
        else:
            return "Waning Crescent"

    # Between Quarter and Full (elongation 90-180°)
    if k > 0.60:
        if waxing:
            return "Waxing Gibbous"
        else:
            return "Waning Gibbous"

    # Fallback (shouldn't normally reach here)
    if waxing:
        return "Waxing Crescent"
    else:
        return "Waning Crescent"


def terminator_angle(jd: float) -> float:
    """Compute the position angle of the Moon's bright limb (terminator).

    This is the angle of the line dividing the illuminated and dark
    portions of the Moon, measured eastward from celestial north.

    Uses the geocentric positions of the Sun and Moon.

    :param jd: Julian Day Number
    :return: Position angle of the bright limb in degrees (0-360)
    """
    sun_ra, sun_dec = moonpos.sun_position(jd)
    moon_ra, moon_dec, _ = moonpos.moon_position(jd)

    # Convert to radians
    s_ra = np.radians(sun_ra)
    s_dec = np.radians(sun_dec)
    m_ra = np.radians(moon_ra)
    m_dec = np.radians(moon_dec)

    # Difference in RA
    d_ra = s_ra - m_ra

    # Position angle of the bright limb
    # chi = arctan2(cos(s_dec) * sin(d_ra),
    #               sin(s_dec) * cos(m_dec) - cos(s_dec) * sin(m_dec) * cos(d_ra))
    x = np.cos(s_dec) * np.sin(d_ra)
    y = (np.sin(s_dec) * np.cos(m_dec)
         - np.cos(s_dec) * np.sin(m_dec) * np.cos(d_ra))

    chi = np.degrees(np.arctan2(x, y))
    return float(chi % 360.0)
