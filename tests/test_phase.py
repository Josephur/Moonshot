"""Tests for the moon.phase module (illumination and phase names)."""

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from moon import phase


class TestIllumination:
    """Illuminated fraction calculations."""

    def test_illumination_range(self):
        """Illumination should always be in [0.0, 1.0]."""
        # Test across many JDs to cover various Moon-Earth-Sun geometries
        jds = [2450000.0 + i * 3.5 for i in range(200)]  # ~2 years of 3.5-day steps
        for jd in jds:
            k = phase.illumination(jd)
            assert 0.0 <= k <= 1.0, f"Illumination {k} out of range at JD {jd}"

    def test_new_moon(self):
        """Near 0 illumination should be close to New Moon (< 0.03)."""
        # Sun and Moon at roughly the same RA (conjunction)
        same_ra = 45.0
        same_dec = 0.0
        sun_pos = (same_ra, same_dec)
        moon_pos = (same_ra + 0.5, same_dec + 0.2, 385000.0)
        k = phase.illumination(0.0, sun_pos=sun_pos, moon_pos=moon_pos)
        assert k < 0.03, f"Expected near-new illumination, got {k}"

    def test_full_moon(self):
        """Near 1.0 illumination should be close to Full Moon (> 0.97)."""
        # Sun and Moon at opposition (RA differs by ~180°)
        sun_pos = (45.0, 0.0)
        moon_pos = (225.0, 0.0, 385000.0)
        k = phase.illumination(0.0, sun_pos=sun_pos, moon_pos=moon_pos)
        assert k > 0.97, f"Expected near-full illumination, got {k}"

    def test_quarter_moon(self):
        """Quarter moon occurs at ~90° elongation, illumination ~ 0.5."""
        sun_pos = (45.0, 0.0)
        moon_pos = (135.0, 0.0, 385000.0)  # 90° separation
        k = phase.illumination(0.0, sun_pos=sun_pos, moon_pos=moon_pos)
        assert 0.45 <= k <= 0.55, f"Expected ~0.5 illumination at quarter, got {k}"


class TestPhaseName:
    """Phase name mapping tests."""

    def test_new_moon(self):
        """Illumination < 0.03 should be 'New Moon'."""
        assert phase.phase_name(0.0, True) == "New Moon"
        assert phase.phase_name(0.01, False) == "New Moon"
        assert phase.phase_name(0.029, True) == "New Moon"

    def test_full_moon(self):
        """Illumination >= 0.97 should be 'Full Moon'."""
        assert phase.phase_name(0.97, True) == "Full Moon"
        assert phase.phase_name(0.99, False) == "Full Moon"
        assert phase.phase_name(1.0, True) == "Full Moon"

    def test_waxing_crescent(self):
        """Waxing crescent: k in [0.03, 0.40), waxing=True."""
        name = phase.phase_name(0.15, waxing=True)
        assert name == "Waxing Crescent"

    def test_waning_crescent(self):
        """Waning crescent: k in [0.03, 0.40), waxing=False."""
        name = phase.phase_name(0.15, waxing=False)
        assert name == "Waning Crescent"

    def test_first_quarter(self):
        """First Quarter: k in [0.40, 0.60], waxing=True."""
        name = phase.phase_name(0.50, waxing=True)
        assert name == "First Quarter"

    def test_last_quarter(self):
        """Last Quarter: k in [0.40, 0.60], waxing=False."""
        name = phase.phase_name(0.50, waxing=False)
        assert name == "Last Quarter"

    def test_waxing_gibbous(self):
        """Waxing Gibbous: k in (0.60, 0.97), waxing=True."""
        name = phase.phase_name(0.80, waxing=True)
        assert name == "Waxing Gibbous"

    def test_waning_gibbous(self):
        """Waning Gibbous: k in (0.60, 0.97), waxing=False."""
        name = phase.phase_name(0.80, waxing=False)
        assert name == "Waning Gibbous"

    def test_phase_name_all(self):
        """All eight phase names should be in the known list."""
        known_phases = {
            "New Moon",
            "Waxing Crescent",
            "First Quarter",
            "Waxing Gibbous",
            "Full Moon",
            "Waning Gibbous",
            "Last Quarter",
            "Waning Crescent",
        }
        test_cases = [
            (0.00, True, "New Moon"),
            (0.10, True, "Waxing Crescent"),
            (0.10, False, "Waning Crescent"),
            (0.50, True, "First Quarter"),
            (0.50, False, "Last Quarter"),
            (0.80, True, "Waxing Gibbous"),
            (0.80, False, "Waning Gibbous"),
            (0.99, True, "Full Moon"),
            (0.99, False, "Full Moon"),
        ]
        for k, waxing, _ in test_cases:
            name = phase.phase_name(k, waxing)
            assert name in known_phases, f"Unknown phase name '{name}' for k={k}"


class TestIsWaxing:
    """Waxing/waning determination."""

    def test_waxing_when_moon_east(self):
        """Moon east of Sun (RA greater) should be waxing."""
        assert phase.is_waxing(45.0, 100.0) is True

    def test_waning_when_moon_west(self):
        """Moon west of Sun (RA smaller) should be waning."""
        assert phase.is_waxing(100.0, 45.0) is False

    def test_waxing_wraparound(self):
        """Waxing detection should handle RA wraparound at 360/0."""
        # Sun near 350°, Moon near 10° (10° ahead = 20° after wrapping)
        assert phase.is_waxing(350.0, 10.0) is True

    def test_waning_wraparound(self):
        """Waning detection should handle RA wraparound at 360/0."""
        # Sun near 10°, Moon near 350° (Moon is behind)
        assert phase.is_waxing(10.0, 350.0) is False


class TestTerminatorAngle:
    """Terminator (bright limb) position angle."""

    def test_terminator_angle_range(self):
        """Terminator angle should be in [0, 360)."""
        for jd in [2450000.0, 2451545.0, 2460000.0]:
            chi = phase.terminator_angle(jd)
            assert 0.0 <= chi < 360.0, f"Terminator angle {chi} out of range at JD {jd}"

    def test_terminator_angle_reproducible(self):
        """Same JD should produce the same terminator angle."""
        chi1 = phase.terminator_angle(2451545.0)
        chi2 = phase.terminator_angle(2451545.0)
        assert chi1 == pytest.approx(chi2)
