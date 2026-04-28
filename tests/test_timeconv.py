"""Tests for the moon.timeconv module (astronomical time conversion)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from moon import timeconv as tc


class TestJulianDate:
    """Julian Day Number calculations."""

    def test_julian_date_known_value(self):
        """JD for 2000-01-01 12:00:00 UTC should equal 2451545.0."""
        jd = tc.julian_date(2000, 1, 1, 12, 0, 0)
        assert jd == pytest.approx(2451545.0, abs=1e-6)

    def test_julian_date_today(self):
        """JD for any modern date should be > 2450000."""
        jd = tc.julian_date(2026, 4, 27, 0, 0, 0)
        assert jd > 2450000.0


class TestSiderealTime:
    """Greenwich and Local Sidereal Time calculations."""

    def test_gmst_range(self):
        """GMST should always be in the range [0, 24) hours."""
        for jd in [2450000.0, 2451545.0, 2460000.0, 2453000.5, 2455000.75]:
            g = tc.gmst(jd)
            assert 0.0 <= g < 24.0, f"GMST {g} out of range for JD {jd}"

    def test_lmst_offset(self):
        """LMST should differ from GMST by longitude / 15 (mod 24)."""
        gmst_hours = 10.0
        lon = -86.2  # Indianapolis
        lst = tc.lmst(gmst_hours, lon)
        expected = (gmst_hours + lon / 15.0) % 24.0
        assert lst == pytest.approx(expected, abs=1e-9)


class TestDeltaT:
    """Delta T (TT - UT1) approximation."""

    def test_delta_t_positive(self):
        """Delta T should be positive for modern years."""
        dt = tc.delta_t(2025.0)
        assert dt > 0.0

    def test_delta_t_reasonable(self):
        """Delta T for 2025 should be around 70 seconds."""
        dt = tc.delta_t(2025.0)
        assert 30.0 < dt < 100.0
