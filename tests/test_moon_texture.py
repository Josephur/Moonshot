"""Unit tests for moon surface texture loading and sampling (moon_texture.py).

Tests cover texture loading, bilinear UV sampling, edge cases, and
fallback behaviour.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from render.moon_texture import load_texture, sample_texture


# ── Texture loading ───────────────────────────────────────────

class TestTextureLoading:
    """Verify the bundled moon texture loads correctly."""

    def test_texture_loads_without_error(self):
        """Texture should load without raising."""
        tex = load_texture()
        assert tex is not None

    def test_texture_is_numpy_array(self):
        """Loaded texture should be a numpy uint8 array."""
        tex = load_texture()
        assert isinstance(tex, np.ndarray)
        assert tex.dtype == np.uint8

    def test_texture_is_3_channel(self):
        """Texture should have 3 colour channels (RGB)."""
        tex = load_texture()
        assert tex.ndim == 3
        assert tex.shape[2] == 3

    def test_texture_has_correct_dimensions(self):
        """Texture should be 512×256 (width×height)."""
        tex = load_texture()
        assert tex.shape[1] == 512, f"Expected width 512, got {tex.shape[1]}"
        assert tex.shape[0] == 256, f"Expected height 256, got {tex.shape[0]}"

    def test_texture_has_valid_pixel_data(self):
        """Pixel values should be in valid uint8 range."""
        tex = load_texture()
        assert tex.min() >= 0
        assert tex.max() <= 255

    def test_texture_not_all_same(self):
        """Texture should not be a solid colour (has actual lunar data)."""
        tex = load_texture()
        assert tex.max() > tex.min(), "Texture appears to be a solid colour"

    def test_texture_is_cached(self):
        """Loading multiple times should return the same object."""
        tex1 = load_texture()
        tex2 = load_texture()
        assert tex1 is tex2

    def test_missing_file_returns_none(self):
        """Loading from a non-existent path should return None."""
        tex = load_texture(path="/tmp/nonexistent_moon.png")
        assert tex is None


# ── UV sampling ───────────────────────────────────────────────

class TestTextureSampling:
    """Verify bilinear-interpolated UV sampling."""

    def setup_method(self):
        self.tex = load_texture()
        assert self.tex is not None, "Texture must load for sampling tests"

    def test_sample_centre(self):
        """Sampling at (0, 0) lon/lat should return valid RGB."""
        r, g, b = sample_texture(0.0, 0.0, self.tex)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_sample_north_pole(self):
        """Sampling at north pole (lat=π/2) should work."""
        r, g, b = sample_texture(0.0, np.pi / 2.0, self.tex)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_sample_south_pole(self):
        """Sampling at south pole (lat=-π/2) should work."""
        r, g, b = sample_texture(0.0, -np.pi / 2.0, self.tex)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_sample_wraps_longitude(self):
        """Sampling at lon=π and lon=-π should give same result (seam)."""
        _, _, b1 = sample_texture(np.pi, 0.0, self.tex)
        _, _, b2 = sample_texture(-np.pi, 0.0, self.tex)
        assert abs(b1 - b2) < 5, (
            f"Seam mismatch at ±π: b={b1} vs b={b2}"
        )

    def test_sample_different_longitudes_differ(self):
        """Different longitudes should (generally) produce different colours."""
        r1, g1, b1 = sample_texture(-np.pi * 0.8, 0.0, self.tex)
        r2, g2, b2 = sample_texture(np.pi * 0.8, 0.0, self.tex)
        # The moon has bright highlands vs dark maria
        # At ±0.8π we're near opposite limbs. They may differ.
        assert (r1, g1, b1) != (r2, g2, b2), (
            "Opposite limbs should have different colours"
        )

    def test_sample_mare_vs_highland(self):
        """Mare (dark) vs highland (bright) should differ."""
        # Mare Tranquillitatis roughly around lon=25°, lat=0°
        mare = sample_texture(np.radians(25), 0.0, self.tex)
        # Highlands roughly on the farside around lon=180°, lat=0°
        highland = sample_texture(np.radians(180), 0.0, self.tex)
        mare_brightness = sum(mare)
        hl_brightness = sum(highland)
        assert hl_brightness > mare_brightness - 100, (
            f"Highlands ({hl_brightness}) should not be much "
            f"darker than mare ({mare_brightness})"
        )

    def test_bilinear_smoothness(self):
        """Neighbouring UV coordinates should give similar results."""
        c1 = sample_texture(0.0, 0.0, self.tex)
        c2 = sample_texture(0.005, 0.005, self.tex)
        diff = sum(abs(c1[i] - c2[i]) for i in range(3))
        assert diff < 30, f"Neighbouring samples differ too much: {diff}"

    def test_returns_tuple_of_ints(self):
        """Return value should be a tuple of three ints."""
        result = sample_texture(0.0, 0.0, self.tex)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, int)


# ── Synthetic texture (unit test) ─────────────────────────────

class TestSamplingOnSyntheticTexture:
    """Verify sampling behaviour with a known synthetic texture."""

    def setup_method(self):
        # Create a small 4×4 checkerboard texture
        self.small_tex = np.array([
            [[255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]],
            [[0, 0, 255], [255, 255, 0], [0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]],
            [[0, 0, 255], [255, 255, 0], [0, 0, 0], [255, 255, 255]],
        ], dtype=np.uint8)

    def test_sample_corner(self):
        """Sample at UV corner should be close to texel [0,0]."""
        # Sample at the absolute corner: u≈0, v≈0 → texel [0,0]
        # u = (lon + π)/(2π) = 0 → lon = -π
        # v = (π/2 − lat)/π = 0 → lat = π/2
        # At the extreme north pole + far west limb, we should get
        # the top-left region of the texture (nev er map coordinate)
        lon = -np.pi
        lat = np.pi / 2.0
        r, g, b = sample_texture(lon, lat, self.small_tex)
        # Should be some valid interpolation of texel [0,0] but may
        # have bilinear bleed from neighbours. Check that red dominates.
        assert r > g + b, f"Expected red-dominant at top-left, got ({r}, {g}, {b})"

    def test_sample_edge_wrap(self):
        """Seam wrapping: u=0 and u=1 should give similar results."""
        r1, g1, b1 = sample_texture(np.pi, 0.0, self.small_tex)
        r2, g2, b2 = sample_texture(-np.pi, 0.0, self.small_tex)
        assert abs(r1 - r2) < 5
        assert abs(g1 - g2) < 5
        assert abs(b1 - b2) < 5
