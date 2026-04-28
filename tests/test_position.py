"""Tests for the moon.position module (sun and moon position calculations).

Uses pytest fixtures and mocks where appropriate. No real network calls.
"""

import sys
import os
from unittest.mock import patch
import numpy as np

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from moon import position as moonpos


class TestSunPosition:
    """Sun position calculations."""

    def test_sun_position(self):
        """Sun RA should be in [0, 360) and Dec in [-90, 90]."""
        for jd in [2450000.0, 2451545.0, 2460000.0, 2453000.0]:
            ra, dec = moonpos.sun_position(jd)
            assert 0.0 <= ra < 360.0, f"RA {ra} out of range for JD {jd}"
            assert -90.0 <= dec <= 90.0, f"Dec {dec} out of range for JD {jd}"

    def test_sun_position_reproducible(self):
        """Same JD should produce the same sun position."""
        ra1, dec1 = moonpos.sun_position(2451545.0)
        ra2, dec2 = moonpos.sun_position(2451545.0)
        assert ra1 == pytest.approx(ra2)
        assert dec1 == pytest.approx(dec2)


class TestMoonPosition:
    """Moon position calculations."""

    def test_moon_alt_az_range(self):
        """Moon altitude should be in [-90, 90], azimuth in [0, 360)."""
        # Test a few JDs and locations
        for lat in [0.0, 39.7, -33.9]:
            for lon in [0.0, -86.2, 151.2]:
                for jd in [2450000.0, 2451545.0, 2460000.0]:
                    alt, az = moonpos.moon_alt_az(lat, lon, jd)
                    assert -90.0 <= alt <= 90.0, (
                        f"Altitude {alt} out of range for "
                        f"lat={lat}, lon={lon}, jd={jd}"
                    )
                    assert 0.0 <= az < 360.0, (
                        f"Azimuth {az} out of range for "
                        f"lat={lat}, lon={lon}, jd={jd}"
                    )

    def test_moon_position_known_date(self):
        """moon_position returns a tuple of (ra, dec, distance)."""
        ra, dec, dist = moonpos.moon_position(2451545.0)
        # RA should be in [0, 360)
        assert 0.0 <= ra < 360.0
        # Dec should be in [-90, 90]
        assert -90.0 <= dec <= 90.0
        # Distance should be roughly 350k-410k km
        assert 300000.0 < dist < 420000.0

    def test_moon_position_reproducible(self):
        """Same JD should produce the same moon position."""
        pos1 = moonpos.moon_position(2451545.0)
        pos2 = moonpos.moon_position(2451545.0)
        assert pos1[0] == pytest.approx(pos2[0])
        assert pos1[1] == pytest.approx(pos2[1])
        assert pos1[2] == pytest.approx(pos2[2])


class TestMoonApparent:
    """Apparent Moon position (with refraction correction)."""

    def test_apparent_alt_az_range(self):
        """Apparent altitude/azimuth should be in valid ranges."""
        # Use mocks to avoid importing atmosphere.refraction, since that
        # requires all atmosphere internals. Instead just test at zenith.
        alt, az = moonpos.moon_apparent_alt_az(0.0, 0.0, 2451545.0)
        assert -90.0 <= alt <= 90.0
        assert 0.0 <= az < 360.0

        # With a higher location (equator, noon-ish JD)
        alt2, az2 = moonpos.moon_apparent_alt_az(39.7, -86.2, 2460000.0)
        assert -90.0 <= alt2 <= 90.0
        assert 0.0 <= az2 < 360.0
