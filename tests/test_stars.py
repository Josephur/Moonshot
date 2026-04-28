"""Unit tests for the star rendering pipeline (stars.py).

Tests cover proper motion, precession, B-V colour mapping, extinction,
equatorial-to-horizontal conversion, and catalog loading.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from render.stars import (
    _airmass,
    _star_intensity,
    _star_size,
    apply_extinction,
    bv_to_rgb,
    equatorial_to_horizontal_vectorised,
    load_catalog,
    precess_j2000_to_epoch,
    proper_motion,
)


# ── Catalog loading ───────────────────────────────────────────

class TestCatalogLoading:
    """Verify the bundled HYG catalog loads and has valid data."""

    def test_loads_without_error(self):
        """Catalog should load without raising."""
        cat = load_catalog()
        assert cat is not None

    def test_has_required_keys(self):
        """Catalog dict should contain all expected arrays."""
        cat = load_catalog()
        for key in ("id", "ra", "dec", "mag", "ci", "pmra", "pmdec"):
            assert key in cat, f"Missing key: {key}"
            assert isinstance(cat[key], np.ndarray)

    def test_has_reasonable_number_of_stars(self):
        """Filtered HYG should have roughly 40K stars (Vmag < 8.0)."""
        cat = load_catalog()
        assert 35000 < len(cat["ra"]) < 50000, (
            f"Unexpected star count: {len(cat['ra'])}"
        )

    def test_catalog_is_cached(self):
        """Loading twice should return the same object (cached)."""
        cat1 = load_catalog()
        cat2 = load_catalog()
        assert cat1 is cat2

    def test_ra_in_decimal_hours(self):
        """RA should be in decimal hours (0-24), not degrees."""
        cat = load_catalog()
        assert cat["ra"].min() >= 0.0
        assert cat["ra"].max() <= 24.0, (
            f"RA max is {cat['ra'].max()}, expected ≤ 24 (decimal hours)"
        )

    def test_dec_in_valid_range(self):
        """Declination should be between -90 and +90 degrees."""
        cat = load_catalog()
        assert cat["dec"].min() >= -90.0
        assert cat["dec"].max() <= 90.0

    def test_magnitudes_under_8(self):
        """All stars should have Vmag < 8.0 (our filter cutoff)."""
        cat = load_catalog()
        assert cat["mag"].max() < 8.0

    def test_ci_contains_nan(self):
        """Some stars should have NaN colour index (missing data)."""
        cat = load_catalog()
        assert np.isnan(cat["ci"]).any(), "Expected some NaN CI values"


# ── Proper motion ─────────────────────────────────────────────

class TestProperMotion:
    """Verify proper-motion correction."""

    def test_no_motion_at_j2000(self):
        """At J2000.0 epoch, proper-motion correction should be zero."""
        ra = np.array([10.0, 100.0, 200.0])
        dec = np.array([20.0, -30.0, 45.0])
        pmra = np.array([100.0, 200.0, 300.0])  # mas/yr
        pmdec = np.array([50.0, -100.0, 150.0])

        ra_out, dec_out = proper_motion(ra, dec, pmra, pmdec, jd=2451545.0)
        np.testing.assert_array_almost_equal(ra_out, ra, decimal=10)
        np.testing.assert_array_almost_equal(dec_out, dec, decimal=10)

    def test_positive_pmra_moves_west(self):
        """Positive pmRA should increase RA when cos(dec) > 0."""
        ra = np.array([100.0])
        dec = np.array([0.0])  # cos = 1
        pmra = np.array([3600000.0])  # 3600000 mas/yr = 1 deg/yr
        pmdec = np.array([0.0])
        # 1 year: 1 deg/yr × 1yr / 3600000 × 3600000 / cos(0) = 1 deg
        # jd = 2451545.0 + 365.25 = 2451910.25 → delta_t = 1.0 yr
        ra_out, _ = proper_motion(ra, dec, pmra, pmdec, jd=2451545.0 + 365.25)
        assert abs(ra_out[0] - 101.0) < 0.1

    def test_proper_motion_preserves_shape(self):
        """Output arrays should have same shape as inputs."""
        ra = np.array([10.0, 20.0, 30.0])
        dec = np.array([0.0, 10.0, 20.0])
        pmra = np.array([100.0, 200.0, 300.0])
        pmdec = np.array([50.0, 60.0, 70.0])
        ra_out, dec_out = proper_motion(ra, dec, pmra, pmdec, jd=2461545.0)
        assert ra_out.shape == ra.shape
        assert dec_out.shape == dec.shape

    def test_zero_proper_motion(self):
        """Zero proper motion should leave coordinates unchanged."""
        ra = np.array([42.0])
        dec = np.array([-15.0])
        pmra = np.array([0.0])
        pmdec = np.array([0.0])
        ra_out, dec_out = proper_motion(ra, dec, pmra, pmdec, jd=2461545.0)
        assert ra_out[0] == pytest.approx(42.0)
        assert dec_out[0] == pytest.approx(-15.0)


# ── Precession ────────────────────────────────────────────────

class TestPrecession:
    """Verify IAU 1976 precession is correctly implemented."""

    def test_no_precession_at_j2000(self):
        """At J2000.0, precession should be identity (no change)."""
        ra = np.array([10.0, 100.0, 200.0])
        dec = np.array([20.0, -30.0, 45.0])
        ra_out, dec_out = precess_j2000_to_epoch(ra, dec, jd=2451545.0)
        np.testing.assert_array_almost_equal(ra_out, ra, decimal=10)
        np.testing.assert_array_almost_equal(dec_out, dec, decimal=10)

    def test_precession_non_zero(self):
        """After 100 years, coordinates should change measurably."""
        ra = np.array([10.0])
        dec = np.array([20.0])
        ra_out, dec_out = precess_j2000_to_epoch(ra, dec, jd=2491545.0)
        assert abs(ra_out[0] - 10.0) > 0.01
        assert abs(dec_out[0] - 20.0) > 0.001

    def test_precession_preserves_shape(self):
        """Output should have same shape as input."""
        ra = np.array([10.0, 20.0, 30.0])
        dec = np.array([0.0, 10.0, 20.0])
        ra_out, dec_out = precess_j2000_to_epoch(ra, dec, jd=2461545.0)
        assert ra_out.shape == ra.shape
        assert dec_out.shape == dec.shape

    def test_precession_returns_valid_ra(self):
        """Precessed RA should be in range [0, 360)."""
        ra = np.array([100.0])
        dec = np.array([45.0])
        ra_out, dec_out = precess_j2000_to_epoch(ra, dec, jd=2461545.0)
        assert 0.0 <= ra_out[0] < 360.0
        assert -90.0 <= dec_out[0] <= 90.0


# ── Equatorial → Horizontal ──────────────────────────────────

class TestEquatorialToHorizontal:
    """Verify alt-az conversion."""

    def test_north_pole_zenith(self):
        """At the north pole, Dec=90° should be at zenith (alt=90)."""
        ra = np.array([0.0])
        dec = np.array([90.0])
        alt, az = equatorial_to_horizontal_vectorised(ra, dec, 90.0, 0.0, 2461545.0)
        assert abs(alt[0] - 90.0) < 0.01

    def test_equator_west(self):
        """Basic sanity: reasonable altitude for a simple case."""
        ra = np.array([30.0])
        dec = np.array([0.0])
        alt, az = equatorial_to_horizontal_vectorised(ra, dec, 0.0, 0.0, 2451545.0)
        assert -90.0 <= alt[0] <= 90.0
        assert 0.0 <= az[0] <= 360.0

    def test_vectorised_output_shape(self):
        """Multiple inputs should produce matching output shapes."""
        ra = np.array([10.0, 50.0, 100.0, 200.0])
        dec = np.array([10.0, 20.0, -30.0, 45.0])
        alt, az = equatorial_to_horizontal_vectorised(ra, dec, 39.77, -86.16, 2461545.0)
        assert alt.shape == (4,)
        assert az.shape == (4,)
        assert np.all(np.isfinite(alt))
        assert np.all(np.isfinite(az))


# ── Atmospheric extinction ────────────────────────────────────

class TestAtmosphericExtinction:
    """Verify extinction calculations."""

    def test_airmass_at_zenith(self):
        """At 90° (zenith), airmass should be ~1.0."""
        am = _airmass(np.array([90.0]))
        assert abs(am[0] - 1.0) < 0.01

    def test_airmass_capped(self):
        """Airmass should be capped at 10.0."""
        am = _airmass(np.array([0.5]))
        assert am[0] <= 10.0

    def test_airmass_vectorised(self):
        """Multiple altitudes should work and increase with lower alt."""
        am = _airmass(np.array([30.0, 60.0, 90.0]))
        assert am.shape == (3,)
        assert am[2] < am[1] < am[0]

    def test_extinction_increases_with_lower_altitude(self):
        """Stars at lower altitude should have greater extinction."""
        mag = np.array([1.0, 1.0])
        alt = np.array([10.0, 40.0])
        ext = apply_extinction(mag, alt)
        assert ext[0] > ext[1]

    def test_extinction_zero_at_zenith(self):
        """At zenith, extinction should be exactly 0.28 mag."""
        mag = np.array([0.0])
        ext = apply_extinction(mag, np.array([90.0]))
        assert abs(ext[0] - 0.28) < 0.01


# ── B-V → RGB colour mapping ─────────────────────────────────

class TestBvToRgb:
    """Verify colour-index to RGB conversion."""

    def test_nan_maps_to_white(self):
        """NaN colour indices should map to white (255, 255, 255)."""
        rgb = bv_to_rgb(np.array([np.nan]))
        assert rgb[0].tolist() == [255, 255, 255]

    def test_negative_blue(self):
        """Negative B-V should be blue-ish."""
        rgb = bv_to_rgb(np.array([-0.4]))
        assert rgb[0, 2] > rgb[0, 0], f"Expected blue channel > red: {rgb[0]}"

    def test_solar_yellow(self):
        """Sun-like B-V (~0.65) should be yellow-ish."""
        rgb = bv_to_rgb(np.array([0.65]))
        r, g, b = rgb[0]
        assert r > b, f"Expected red > blue for yellow star: ({r}, {g}, {b})"
        assert abs(r - g) < 50, "R and G should be similar for yellow"

    def test_red_star(self):
        """High B-V (> 1.0) should be red-ish."""
        rgb = bv_to_rgb(np.array([1.5]))
        assert rgb[0, 0] > rgb[0, 2], f"Expected red > blue: {rgb[0]}"

    def test_vectorised_output(self):
        """Multiple B-V values should produce (N, 3) output."""
        rgb = bv_to_rgb(np.array([0.0, 0.5, 1.0, np.nan]))
        assert rgb.shape == (4, 3)

    def test_rgb_in_range(self):
        """All RGB values should be in [0, 255]."""
        rgb = bv_to_rgb(np.array([-0.5, 0.0, 0.5, 1.0, 1.5, np.nan]))
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 255)


# ── Star properties from magnitude ────────────────────────────

class TestStarProperties:
    """Verify magnitude-to-size and magnitude-to-intensity mapping."""

    def test_brightest_stars_largest(self):
        """Mag < 0.5 should be size 3."""
        sizes = _star_size(np.array([-1.5, 0.0, 0.3]))
        assert np.all(sizes == 3)

    def test_mid_stars_size_2(self):
        """Mag 0.5-2.5 should be size 2."""
        sizes = _star_size(np.array([0.5, 1.0, 2.0, 2.49]))
        assert np.all(sizes == 2)

    def test_dim_stars_size_1(self):
        """Mag >= 2.5 should be size 1."""
        sizes = _star_size(np.array([2.5, 5.0, 7.9]))
        assert np.all(sizes == 1)

    def test_intensity_decreases_with_magnitude(self):
        """Brighter (lower mag) stars should have higher intensity."""
        i1 = _star_intensity(np.array([0.0]))
        i5 = _star_intensity(np.array([5.0]))
        assert i5[0] < i1[0]

    def test_sirius_intensity(self):
        """Sirius (mag -1.44) should have intensity near 1.0."""
        i = _star_intensity(np.array([-1.44]))
        assert i[0] > 0.9, f"Sirius intensity too low: {i[0]}"

    def test_star_sizes_vectorised(self):
        """Multiple magnitudes should produce same-size output."""
        sizes = _star_size(np.array([-1.0, 1.0, 5.0, 7.0]))
        assert sizes.shape == (4,)
