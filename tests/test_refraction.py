"""Tests for the atmosphere.refraction module (Saemundsson formula)."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atmosphere import refraction


class TestRefractionCorrection:
    """Direct refraction correction (apparent -> correction)."""

    def test_zero_refraction_at_zenith(self):
        """At 90° altitude (zenith), refraction should be ~0."""
        r = refraction.refraction_correction(90.0)
        assert r == pytest.approx(0.0, abs=1e-3)

    def test_positive_refraction(self):
        """Refraction correction should always be positive above horizon."""
        for alt in [1.0, 5.0, 10.0, 30.0, 45.0, 60.0, 85.0]:
            r = refraction.refraction_correction(alt)
            assert r >= 0.0, f"Negative refraction {r} at altitude {alt}"

    def test_refraction_increases_near_horizon(self):
        """Refraction should be larger near 0° than at higher altitudes."""
        r_horizon = refraction.refraction_correction(0.5)
        r_mid = refraction.refraction_correction(45.0)
        assert r_horizon > r_mid, (
            f"Refraction at horizon ({r_horizon}) should be > "
            f"refraction at 45° ({r_mid})"
        )

    def test_refraction_very_near_horizon(self):
        """Refraction at 1° should be more than at 5°."""
        r_1deg = refraction.refraction_correction(1.0)
        r_5deg = refraction.refraction_correction(5.0)
        assert r_1deg > r_5deg

    def test_refraction_pressure_scaling(self):
        """Higher pressure should increase refraction correction."""
        r_low = refraction.refraction_correction(10.0, pressure_mbar=800.0)
        r_high = refraction.refraction_correction(10.0, pressure_mbar=1013.0)
        assert r_high > r_low

    def test_refraction_temperature_scaling(self):
        """Higher temperature should decrease refraction correction."""
        r_cold = refraction.refraction_correction(10.0, temp_c=-10.0)
        r_hot = refraction.refraction_correction(10.0, temp_c=40.0)
        assert r_cold > r_hot


class TestApparentFromTrue:
    """True -> apparent altitude conversion."""

    def test_apparent_above_true(self):
        """Apparent altitude should be above true altitude for positive alts."""
        for true_alt in [1.0, 5.0, 15.0, 45.0]:
            app_alt = refraction.apparent_from_true(true_alt)
            assert app_alt > true_alt, (
                f"Apparent alt {app_alt} should be > true alt {true_alt}"
            )

    def test_apparent_near_zenith(self):
        """Near zenith, apparent should be nearly equal to true."""
        app_alt = refraction.apparent_from_true(89.0)
        assert app_alt == pytest.approx(89.0, abs=0.01)

    def test_apparent_below_horizon(self):
        """Below -5°, apparent should equal true (negligible refraction)."""
        app_alt = refraction.apparent_from_true(-10.0)
        assert app_alt == pytest.approx(-10.0, abs=1e-9)

    def test_apparent_from_true_reasonable(self):
        """At 1° true altitude, apparent should be 1° + reasonable correction."""
        app_alt = refraction.apparent_from_true(1.0)
        assert 1.0 < app_alt < 2.0
