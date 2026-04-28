"""Tests for the atmosphere.scattering module (moon color tint)."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atmosphere import scattering


class TestAirMass:
    """Air mass calculations (from atmosphere.airmass, tested via scattering)."""

    def test_air_mass_zenith(self):
        """At zenith (90°), air mass should be approximately 1.0."""
        from atmosphere import airmass
        am = airmass.air_mass(90.0)
        assert am == pytest.approx(1.0, abs=0.01)

    def test_air_mass_increases(self):
        """Lower altitude should produce higher air mass."""
        from atmosphere import airmass
        am_high = airmass.air_mass(60.0)
        am_low = airmass.air_mass(10.0)
        assert am_low > am_high, (
            f"Air mass at 10° ({am_low}) should be > air mass at 60° ({am_high})"
        )

    def test_air_mass_inf_at_horizon(self):
        """At or below horizon, air mass should be infinite."""
        from atmosphere import airmass
        assert airmass.air_mass(0.0) == float('inf')
        assert airmass.air_mass(-5.0) == float('inf')


class TestMoonColorTint:
    """Moon color tint calculations (RGB multipliers)."""

    def test_moon_color_white_at_zenith(self):
        """At high altitude (85°), RGB multipliers should be close to white (~0.95+)."""
        r, g, b = scattering.moon_color_tint(85.0)
        # All channels should be near 1.0 at zenith
        assert r > 0.90, f"Red multiplier too low at zenith: {r}"
        assert g > 0.90, f"Green multiplier too low at zenith: {g}"
        assert b > 0.90, f"Blue multiplier too low at zenith: {b}"

    def test_color_redder_at_horizon(self):
        """At low altitude, the moon should be redder (more attenuation in blue)."""
        r_high, g_high, b_high = scattering.moon_color_tint(60.0)
        r_low, g_low, b_low = scattering.moon_color_tint(5.0)
        # Red channel should be attenuated less than blue at low altitude
        # (the ratio r/b should be higher at low altitude)
        ratio_high = r_high / max(b_high, 1e-10)
        ratio_low = r_low / max(b_low, 1e-10)
        assert ratio_low > ratio_high, (
            f"R/B ratio should be higher at low altitude "
            f"({ratio_low} vs {ratio_high})"
        )

    def test_zero_altitude_returns_black(self):
        """At 0° altitude, the moon should return black (0, 0, 0)."""
        r, g, b = scattering.moon_color_tint(0.0)
        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

    def test_moon_color_range(self):
        """RGB multipliers should always be in [0, 1]."""
        for alt in [1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 85.0]:
            r, g, b = scattering.moon_color_tint(alt)
            assert 0.0 <= r <= 1.0, f"R={r} out of range at alt={alt}"
            assert 0.0 <= g <= 1.0, f"G={g} out of range at alt={alt}"
            assert 0.0 <= b <= 1.0, f"B={b} out of range at alt={alt}"

    def test_humidity_reduces_brightness(self):
        """Higher humidity should dim all channels (increase scattering)."""
        r_dry, g_dry, b_dry = scattering.moon_color_tint(30.0, humidity_pct=10.0)
        r_humid, g_humid, b_humid = scattering.moon_color_tint(30.0, humidity_pct=90.0)
        assert r_humid <= r_dry + 1e-9
        assert g_humid <= g_dry + 1e-9
        assert b_humid <= b_dry + 1e-9

    def test_reproducible(self):
        """Same inputs should produce the same tint."""
        t1 = scattering.moon_color_tint(30.0, temp_c=15.0, pressure_mbar=1013.0, humidity_pct=50.0)
        t2 = scattering.moon_color_tint(30.0, temp_c=15.0, pressure_mbar=1013.0, humidity_pct=50.0)
        assert t1 == t2
