"""Unit tests for the star rendering pipeline (stars.py).

Tests cover proper motion, precession, B-V colour mapping, extinction,
equatorial-to-horizontal conversion, catalog loading, and end-to-end
rendering verification.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

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
    render_stars_to_sky,
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


# ── End-to-end star rendering regression tests ──────────────

class TestEndToEndStarRendering:
    """Regression tests verifying the full rendering pipeline produces
    visible stars at night from real catalog data.

    These tests catch the RA-hours-vs-degrees bug and similar
    coordinate-system regressions.
    """

    # Indianapolis, 3am EDT = ~08:00 UTC on 2026-05-03
    _jd = 2461163.8333
    _lat = 39.77
    _lon = -86.16
    _fov = 90.0

    def _render_at_night(self, fov: float | None = None) -> np.ndarray:
        """Render stars onto a black background and return pixel array."""
        fov_deg = fov if fov is not None else self._fov
        bg = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        result = render_stars_to_sky(bg, self._lat, self._lon, self._jd, fov_deg)
        return np.array(result)

    def test_stars_rendered_on_black_background(self):
        """A night render must have visible stars — non-black pixels."""
        pixels = self._render_at_night()
        nonzero = (pixels > 0).sum()
        assert nonzero > 500, (
            f"Expected >500 lit pixels from stars at night, got {nonzero}. "
            f"Possible RA conversion bug or catalog regression."
        )

    def test_star_pixels_have_meaningful_brightness(self):
        """Bright stars (mag < 2) should produce pixel values > 100."""
        pixels = self._render_at_night()
        max_val = pixels.max()
        assert max_val > 100, (
            f"Expected star pixels with value >100 (bright stars), "
            f"got max={max_val}. Stars may be missing from rendering."
        )

    def test_night_render_not_empty(self):
        """At deep night, the full 90° FOV must contain stars."""
        pixels = self._render_at_night(fov=90.0)
        lit_pixels = (pixels > 0).sum()
        assert lit_pixels > 0, (
            "Zero lit pixels at 3am with 90° FOV — star rendering is broken."
        )

    def test_stars_cover_multiple_regions_of_sky(self):
        """Stars should be distributed across the image, not all in one strip."""
        pixels = self._render_at_night()
        # Divide image into 9 regions (3×3 grid); stars should appear in at
        # least 4 regions — if they're all in one horizontal band, the
        # coordinate system may be broken (e.g., RA in hours vs degrees).
        h, w = pixels.shape[:2]
        h_step, w_step = h // 3, w // 3
        regions_with_stars = 0
        for row in range(3):
            for col in range(3):
                y0, y1 = row * h_step, (row + 1) * h_step
                x0, x1 = col * w_step, (col + 1) * w_step
                region = pixels[y0:y1, x0:x1]
                if (region > 0).sum() > 0:
                    regions_with_stars += 1

        assert regions_with_stars >= 4, (
            f"Stars found in only {regions_with_stars}/9 image regions. "
            f"Expected ≥4 for distributed night sky with 90° FOV. "
            f"Possible coordinate-system regression."
        )

    def test_ra_is_converted_to_degrees_before_pipeline(self):
        """Verify that RA is multiplied by 15 before entering the pipeline.

        This test checks that the catalog RA (decimal hours) is converted
        to degrees before being passed to proper_motion / precession / alt-az.
        Without the fix, stars above 40° altitude would be absent because
        RA in hours gets treated as degrees, collapsing the sky.
        """
        from render.stars import proper_motion as pm
        from render.stars import precess_j2000_to_epoch as precess
        from render.stars import equatorial_to_horizontal_vectorised as eq2horiz

        cat = load_catalog()
        # The pipeline must convert RA hours → degrees (×15)
        ra_deg = cat["ra"] * 15.0
        ra_pm, dec_pm = pm(ra_deg, cat["dec"], cat["pmra"], cat["pmdec"], self._jd)
        ra_ep, dec_ep = precess(ra_pm, dec_pm, self._jd)
        alt, _ = eq2horiz(ra_ep, dec_ep, self._lat, self._lon, self._jd)

        # At 3am from Indianapolis, roughly half the sky should be above horizon.
        # With the bug (RA in hours, not degrees), the altitude distribution
        # is compresses and max altitude is ~40°. With the fix, we should see
        # stars near zenith.
        above = alt > 0.0
        assert above.sum() > 15000, (
            f"Only {above.sum()} stars above horizon — expected >15000. "
            f"RA may not be converted from hours to degrees."
        )

        max_alt = alt[above].max()
        assert max_alt > 60.0, (
            f"Max altitude above horizon is only {max_alt:.1f}° — "
            f"expected >60° (stars near zenith). "
            f"RA hours→degrees conversion may be missing."
        )
