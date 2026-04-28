"""Visual regression tests using local Ollama + moondream vision model.

These tests render Moonshot images with known parameters and use the
vision model to verify expected visual features are present.

Skipped automatically if Ollama / moondream is not available.
"""

import pytest

from tests.conftest import (
    ask_with_retry,
    check_yes,
    extract_choice,
    ollama_not_ready,
    render_image,
)

pytestmark = ollama_not_ready


# ── V1: Moon Visibility ─────────────────────────────────────
class TestMoonVisibility:
    """V1: The moon should be clearly visible."""

    def test_moon_visible(self):
        """Moon visible at FOV=10."""
        path = render_image("v1_moon.png", lat=39.77, lon=-86.16,
                            date="2026-04-27", time="21:00", fov=10)
        response = ask_with_retry(path, "Is there a moon visible in this image?")
        assert check_yes(response), f"Moon not detected.\nModel: {response!r}"


# ── V2: Horizon ───────────────────────────────────────────────
class TestHorizon:
    """V2: Horizon silhouette should always be present."""

    @pytest.mark.xfail(reason="moondream can't resolve horizon/moon contrast at center")
    def test_horizon_present(self):
        """Horizon (land/terrain at bottom) should be visible."""
        path = render_image("v2_horizon.png")
        response = ask_with_retry(path, "Is there a dark ground or shoreline at the bottom of this image?")
        assert check_yes(response), f"Horizon not detected.\nModel: {response!r}"


# ── V3: Sky Type (Night/Twilight) ─────────────────────────────
class TestSkyType:
    """V3: 9pm EDT should produce twilight or night sky."""

    def test_sky_is_night_or_twilight(self):
        """Sky should be described as night/dark at 9pm."""
        path = render_image("v3_skytype.png", time="21:00")
        response = ask_with_retry(path, "Describe the sky color at the top of this image.")
        match = extract_choice(response, {"DARK", "BLACK", "BLUE", "NIGHT"})
        assert match is not None, f"Sky not night-like.\nModel: {response!r}"


# ── V4: Annotations ────────────────────────────────────────
class TestAnnotations:
    """V4: Data overlay text should be visible."""

    @pytest.mark.xfail(reason="moondream can't resolve small annotation text")
    def test_annotations_present(self):
        """Text/numbers should be visible on the image."""
        path = render_image("v4_annotations.png")
        response = ask_with_retry(path, "Is there any text or numbers written on this image?")
        assert check_yes(response), f"Annotations not detected.\nModel: {response!r}"


# ── V5: Stars ──────────────────────────────────────────────
class TestStars:
    """V5: Deep night (3am) should show stars."""

    @pytest.mark.xfail(reason="moondream misses stars when large moon is present")
    def test_stars_visible_deep_night(self):
        """Stars should be visible at 3am."""
        path = render_image("v5_stars.png", time="03:00")
        response = ask_with_retry(path, "Are there stars or small white dots visible in the dark sky?")
        assert check_yes(response), f"Stars not detected.\nModel: {response!r}"


# ── V6: Sky Color Sanity (Wrong Sky Detection) ─────────────
class TestWrongSkyDetection:
    """V6: Negative tests — model must detect contradictions."""

    def test_daytime_sky_at_noon(self):
        """Noon render should be daytime."""
        path = render_image("v6a_noon.png", time="12:00")
        response = ask_with_retry(path, "Describe the sky in this image. Is it bright and daytime or dark and night?")
        assert "DAY" in response.upper() or "BRIGHT" in response.upper(), \
            f"Noon sky not daytime.\nModel: {response!r}"

    def test_daytime_sky_not_at_3am(self):
        """3am render should NOT be daytime."""
        path = render_image("v6b_3am_not_daytime.png", time="03:00")
        response = ask_with_retry(path, "Is this a daytime or night scene?")
        assert "NIGHT" in response.upper() or "DARK" in response.upper(), \
            f"3am incorrectly called daytime.\nModel: {response!r}"

    @pytest.mark.xfail(reason="moondream sees noon scene as 'sunrise/sunset' due to warm scattering tint")
    def test_night_sky_not_at_noon(self):
        """Noon render should NOT be night."""
        path = render_image("v6c_noon_not_night.png", time="12:00")
        response = ask_with_retry(path, "Is this a daytime or night scene?")
        assert "DAY" in response.upper() or "BRIGHT" in response.upper(), \
            f"Noon incorrectly called night.\nModel: {response!r}"
